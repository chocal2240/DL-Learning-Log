"""Beam-search decoding for the Transformer model."""

from __future__ import annotations

import torch

from tools.data_loader import subsequent_mask


class Beam:
    """Keep the partial hypotheses for one source sentence."""

    def __init__(self, size: int, pad: int, bos: int, eos: int, device: torch.device) -> None:
        self.size = size
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.device = device
        self._done = False

        self.scores = torch.zeros(size, dtype=torch.float, device=device)
        self.all_scores: list[torch.Tensor] = []
        self.prev_ks: list[torch.Tensor] = []
        self.next_ys = [torch.full((size,), pad, dtype=torch.long, device=device)]
        self.next_ys[0][0] = bos

    @property
    def done(self) -> bool:
        return self._done

    def get_current_state(self) -> torch.Tensor:
        """Return all current hypotheses as a ``beam_size x sequence_len`` tensor."""

        return self.get_tentative_hypothesis()

    def get_current_origin(self) -> torch.Tensor:
        return self.prev_ks[-1]

    def advance(self, word_logprob: torch.Tensor) -> bool:
        """Update the beam with next-token log-probabilities."""

        vocab_size = word_logprob.size(1)
        if self.prev_ks:
            beam_scores = word_logprob + self.scores.unsqueeze(1)
        else:
            beam_scores = word_logprob[0]

        flat_scores = beam_scores.reshape(-1)
        best_scores, best_ids = flat_scores.topk(self.size, dim=0, largest=True, sorted=True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        prev_k = best_ids // vocab_size
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_ids - prev_k * vocab_size)

        if self.next_ys[-1][0].item() == self.eos:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sort(self.scores, dim=0, descending=True)

    def get_the_best_score_and_idx(self) -> tuple[torch.Tensor, torch.Tensor]:
        scores, ids = self.sort_scores()
        return scores[0], ids[0]

    def get_tentative_hypothesis(self) -> torch.Tensor:
        if len(self.next_ys) == 1:
            return self.next_ys[0].unsqueeze(1)

        _, keys = self.sort_scores()
        hypotheses = [[self.bos, *self.get_hypothesis(key)] for key in keys]
        return torch.tensor(hypotheses, dtype=torch.long, device=self.device)

    def get_hypothesis(self, beam_index: torch.Tensor | int) -> list[int]:
        """Walk back through beam pointers to build a full hypothesis."""

        index = int(beam_index.item() if isinstance(beam_index, torch.Tensor) else beam_index)
        hypothesis: list[int] = []

        for step in range(len(self.prev_ks) - 1, -1, -1):
            hypothesis.append(int(self.next_ys[step + 1][index].item()))
            index = int(self.prev_ks[step][index].item())

        return hypothesis[::-1]


def beam_search(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    pad: int,
    bos: int,
    eos: int,
    beam_size: int,
    device: torch.device,
) -> tuple[list[list[list[int]]], list[torch.Tensor]]:
    """Decode a batch with beam search."""

    def get_inst_idx_to_tensor_position_map(inst_idx_list: list[int]) -> dict[int, int]:
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(
        beamed_tensor: torch.Tensor,
        current_active_positions: torch.Tensor,
        previous_active_count: int,
        beam_count: int,
    ) -> torch.Tensor:
        _, *remaining_shape = beamed_tensor.size()
        active_count = len(current_active_positions)
        new_shape = (active_count * beam_count, *remaining_shape)
        beamed_tensor = beamed_tensor.view(previous_active_count, -1)
        beamed_tensor = beamed_tensor.index_select(0, current_active_positions)
        return beamed_tensor.view(*new_shape)

    def collate_active_info(
        src_enc: torch.Tensor,
        active_src_mask: torch.Tensor,
        inst_idx_to_position_map: dict[int, int],
        active_inst_idx_list: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
        previous_active_count = len(inst_idx_to_position_map)
        active_positions = [inst_idx_to_position_map[index] for index in active_inst_idx_list]
        active_positions_tensor = torch.tensor(active_positions, dtype=torch.long, device=device)

        src_enc = collect_active_part(src_enc, active_positions_tensor, previous_active_count, beam_size)
        active_src_mask = collect_active_part(
            active_src_mask,
            active_positions_tensor,
            previous_active_count,
            beam_size,
        )
        return src_enc, active_src_mask, get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    def prepare_beam_dec_seq(inst_dec_beams: list[Beam], len_dec_seq: int) -> torch.Tensor:
        partial_sequences = [beam.get_current_state() for beam in inst_dec_beams if not beam.done]
        partial_sequences = torch.stack(partial_sequences).to(device)
        return partial_sequences.view(-1, len_dec_seq)

    def predict_word(
        dec_seq: torch.Tensor,
        enc_output: torch.Tensor,
        active_src_mask: torch.Tensor,
        active_instance_count: int,
        beam_count: int,
    ) -> torch.Tensor:
        tgt_mask = subsequent_mask(dec_seq.size(1)).to(device=dec_seq.device)
        decoder_output = model.decode(enc_output, active_src_mask, dec_seq, tgt_mask)
        word_logprob = model.generator(decoder_output[:, -1])
        return word_logprob.view(active_instance_count, beam_count, -1)

    def collect_active_inst_idx_list(
        inst_beams: list[Beam],
        word_prob: torch.Tensor,
        inst_idx_to_position_map: dict[int, int],
    ) -> list[int]:
        active_inst_idx_list: list[int] = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            if not is_complete:
                active_inst_idx_list.append(inst_idx)
        return active_inst_idx_list

    def beam_decode_step(
        inst_dec_beams: list[Beam],
        len_dec_seq: int,
        enc_output: torch.Tensor,
        active_src_mask: torch.Tensor,
        inst_idx_to_position_map: dict[int, int],
        beam_count: int,
    ) -> list[int]:
        active_instance_count = len(inst_idx_to_position_map)
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        word_logprob = predict_word(
            dec_seq,
            enc_output,
            active_src_mask,
            active_instance_count,
            beam_count,
        )
        return collect_active_inst_idx_list(inst_dec_beams, word_logprob, inst_idx_to_position_map)

    def collect_hypothesis_and_scores(
        inst_dec_beams: list[Beam],
        n_best: int,
    ) -> tuple[list[list[list[int]]], list[torch.Tensor]]:
        all_hypotheses: list[list[list[int]]] = []
        all_scores: list[torch.Tensor] = []

        for beam in inst_dec_beams:
            scores, tail_indices = beam.sort_scores()
            all_scores.append(scores[:n_best])
            all_hypotheses.append([beam.get_hypothesis(index) for index in tail_indices[:n_best]])

        return all_hypotheses, all_scores

    with torch.no_grad():
        src_enc = model.encode(src, src_mask)

        batch_size, sentence_len, hidden_dim = src_enc.size()
        src_enc = src_enc.repeat(1, beam_size, 1).view(batch_size * beam_size, sentence_len, hidden_dim)
        src_mask = src_mask.repeat(1, beam_size, 1).view(batch_size * beam_size, 1, src_mask.shape[-1])

        inst_dec_beams = [Beam(beam_size, pad, bos, eos, device) for _ in range(batch_size)]
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        for len_dec_seq in range(1, max_len + 1):
            active_inst_idx_list = beam_decode_step(
                inst_dec_beams,
                len_dec_seq,
                src_enc,
                src_mask,
                inst_idx_to_position_map,
                beam_size,
            )

            if not active_inst_idx_list:
                break

            src_enc, src_mask, inst_idx_to_position_map = collate_active_info(
                src_enc,
                src_mask,
                inst_idx_to_position_map,
                active_inst_idx_list,
            )

    return collect_hypothesis_and_scores(inst_dec_beams, beam_size)
