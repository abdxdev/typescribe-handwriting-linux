import logging
import os

import numpy as np

from handwriting_synthesis import drawing
from handwriting_synthesis.config import prediction_path, checkpoint_path, style_path
from handwriting_synthesis.hand._draw import _draw, page_line_pieces, unpack_page
from handwriting_synthesis.rnn import RNN


class Hand(object):
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = RNN(
            log_dir='logs',
            checkpoint_dir=checkpoint_path,
            prediction_dir=prediction_path,
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    @staticmethod
    def _batches(total, first_chunk, chunk_lines):
        """(start, stop) line ranges to sample together; one range when not chunking."""
        if not chunk_lines:
            return [(0, total)]
        batches, start, size = [], 0, first_chunk or chunk_lines
        while start < total:
            batches.append((start, min(total, start + size)))
            start += size
            size = chunk_lines
        return batches

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None, page=None,
              pressure=0.0, on_line=None, first_chunk=None, chunk_lines=None):
        """Write `lines` as handwriting to an SVG page (plus a PNG).

        With `chunk_lines` the lines are sampled a few at a time instead of all at once
        (a small `first_chunk` gets ink on screen sooner). After each chunk
        `on_line(index, pieces, color)` is called for every line that has ink, where
        `pieces` is a list of (points, width) in page units. Chunking only changes
        when ink becomes available, not how it looks.
        """
        valid_char_set = set(drawing.alphabet)
        
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        count = len(lines)
        stroke_colors = stroke_colors or ['black'] * count
        stroke_widths = stroke_widths or [2] * count
        limit = unpack_page(page)["total_lines"]

        drawn_lines = ['' if line == '.' else line for line in lines]
        pieces = [None] * count
        for start, stop in self._batches(count, first_chunk, chunk_lines):
            # Blank lines and lines past the ruled page are never drawn, so don't sample them.
            batch = [i for i in range(start, stop) if drawn_lines[i] and i < limit]
            if batch:
                strokes = self._sample(
                    [lines[i] for i in batch],
                    biases=None if biases is None else [biases[i] for i in batch],
                    styles=None if styles is None else [styles[i] for i in batch],
                )
                for i, offsets in zip(batch, strokes):
                    pieces[i] = page_line_pieces(offsets, i, page, stroke_widths[i], pressure)
                    if on_line:
                        on_line(i, pieces[i], stroke_colors[i])

        _draw(None, drawn_lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths, page=page,
              pressure=pressure, pieces=pieces)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40 * max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load(f"{style_path}/style-{style}-strokes.npy")
                c_p = np.load(f"{style_path}/style-{style}-chars.npy").tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples
