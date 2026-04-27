# implemented by RZ
# profile the experiment using tensorboard

from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
from implementation import code_manipulation
from torch.utils.tensorboard import SummaryWriter


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        # self._pkl_dir = pkl_dir
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        # Monotonic write counter — every successful disk write bumps this
        # by 1 and the resulting file name is `sample_{seq:06d}.json`. We
        # used to name files after `program.global_sample_nums`, which is
        # not actually unique across reflector triage / island reset code
        # paths and silently overwrote previously-written samples — log
        # and disk diverged, bench_heuristic picked phantom "best" rows.
        # A dedicated profiler-owned counter makes overwrites physically
        # impossible regardless of what global_sample_nums is reused for.
        self._write_seq: int = 0

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )

    def _write_json(self, programs: code_manipulation.Function):
        """Write `programs` to a *fresh*, monotonically-numbered file.

        File naming is decoupled from `program.global_sample_nums` on
        purpose — see the long comment in `__init__`. The original
        sample order is still recorded inside the JSON payload so
        downstream tools (summarize_run, bench_heuristic) can recover
        the chronological view if they want to.
        """
        self._write_seq += 1
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(programs)
        score = programs.score
        content = {
            'write_seq': self._write_seq,
            'sample_order': sample_order,
            'function': function_str,
            'score': score,
            'thought': programs.thought,
        }
        path = os.path.join(
            self._json_dir, f'sample_{self._write_seq:06d}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def register_function(self, programs: code_manipulation.Function):
        """Record a sample. Always writes to disk; only the first time a
        given `sample_orders` is seen do we also bump counters and emit
        the verbose stdout block — that keeps log readability identical
        to the previous behaviour while making the disk source of truth.

        The dict-based de-dup used to gate `_write_json` too, which is
        why two programs with the same `sample_orders` (e.g. main path
        vs. reflector triage retry) silently overwrote each other on
        disk. We now write every call and rely on the unique
        `_write_seq` filename to keep them apart.
        """
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
        # ALWAYS persist to disk, even on duplicate sample_orders, so
        # the on-disk trajectory is complete and `find_best_sample_in_dir`
        # has every candidate to compare. Filenames come from `_write_seq`,
        # not from `sample_orders`, so collisions are impossible.
        self._write_json(programs)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        # function_name = function.name
        # function_body = function.body.strip('\n')
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        thought = getattr(function, 'thought', None)
        print(f'================= Evaluated Function =================')
        if thought:
            print(f'Thought      : {thought}')
            print(f'------------------------------------------------------')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

        # update ...
        # self._each_sample_best_program_score.append(self._cur_best_program_score)
        # self._each_sample_evaluate_success_program_num.append(self._evaluate_success_program_num)
        # self._each_sample_evaluate_failed_program_num.append(self._evaluate_failed_program_num)
        # self._each_sample_tot_sample_time.append(self._tot_sample_time)
        # self._each_sample_tot_evaluate_time.append(self._tot_evaluate_time)
