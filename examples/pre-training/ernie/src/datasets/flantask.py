# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import io
import random
import logging
import json
import itertools
import gzip
from glob import glob

import paddle

logger = logging.getLogger(__name__)


class Processor:
    def __init__(self, *args, **kwargs):
        """
        构造函数，可以传入任意字段，运行时构造函数的输入由一个json提供(task_spec.json)。
        """
        pass

    def __call__(self, line):
        """
        处理函数，对一**行**数据的。如果无法解析则返回None，否则返回两个字符串：`return 'source', 'target'`
        """
        if 1 / 0:
            return
        return "encoder 输入文本", "decoder 输入文本"


class JsonProcessor:
    def __init__(self, tokenizer, prompt, target_prompt, encoding="utf-8", **kwargs):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.target_prompt = target_prompt
        self.encoding = encoding

    def __call__(self, line):
        try:
            jdict = json.loads(line)
        except Exception as e:
            logger.exception(e)
            logger.warn(f"Bad Json Data data:{line}, skip")
            return
        jdict.update(
            mask=self.tokenizer.mask_token,
            bos=self.tokenizer.bos_token,
            eos=self.tokenizer.eos_token,
        )
        src = self.prompt.format(**jdict)
        tgt = self.target_prompt.format(**jdict)
        if isinstance(tgt, list):
            tgt = "\n".join(tgt)
        src = src.replace("\n", "[unused88]")
        tgt = tgt.replace("\n", "[unused88]")
        return src, tgt


class TsvProcessor:
    def __init__(self, tokenizer, prompt, target_prompt, encoding="utf-8", **kwargs):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.target_prompt = target_prompt
        self.encoding = encoding

    def __call__(self, line):
        try:
            lines = line.rstrip("\n").split("\t")
            jdict = dict(zip([f"slot{i}" for i in range(len(lines))], lines))
        except Exception as e:
            logger.exception(e)
            logger.warn(f"Bad Json Data data:{line}, skip")
            return
        jdict.update(mask=self.tokenizer.mask_token)
        src = self.prompt.format(**jdict)
        tgt = self.target_prompt.format(**jdict)
        if isinstance(tgt, list):
            tgt = "\n".join(tgt)
        return src, tgt


def make_reader(
    filelist,
    tokenizer,
    task_processor,
    rng,
    seqlen,
    dp_size=1,
    dp_rank=0,
    is_decoder_only=False,
    no_pad=False,
):
    if not is_decoder_only:
        assert not no_pad, "t5 should not use no-pad"

    def _tokenize(txt, eos_id=None, random_truncate=True):
        ids = tokenizer.encode(
            txt,
            add_special_tokens=False,
        )
        if len(ids) > seqlen:
            pos = ids.index(tokenizer.mask_token_id)
            s = min(pos + seqlen, len(ids))
            random_offset = (
                rng.randint(0, s - pos) if random_truncate else 0
            )  # shoud we do random?
            s -= random_offset
            ids = ids[s - seqlen : s]
        if eos_id is not None:
            ids = ids + [eos_id]
        return paddle.to_tensor(ids)

    def make_encoder_decoder_data(src, tgt):
        ids = _tokenize(
            src,
        )
        if tokenizer.mask_token_id not in ids:
            logger.warn(f"mask token truncated, ignore this case: {src}")
            return None
        mask = (ids != tokenizer.ignored_index).astype("int32")
        if tokenizer.mask_token not in tgt:
            tgt = tokenizer.mask_token + tgt
            eos_id = tokenizer.mask_token_id + 1
        else:
            eos_id = None
        label = _tokenize(tgt, eos_id=eos_id)
        ret = dict(
            input_ids=ids,
            attention_mask=mask,
            labels=label,
        )
        return ret

    def _tokenize_decoder_only(
        src,
        tgt,
        seqlen,
    ):
        ret = tokenizer(
            src,
            tgt,
        )
        ids = ret.pop("input_ids")
        sids = ret.pop("token_type_ids")
        assert len(ids) == len(sids), (len(ids), len(sids))

        if len(ids) >= seqlen:
            c = seqlen
            pos = rng.randint(int(-c / 4), len(ids) - int(c / 4))  # from gopher
            s = max(0, pos)
            e = min(len(ids), pos + c)
            ids = ids[s:e]
            sids = sids[s:e]
        return paddle.to_tensor(ids), paddle.to_tensor(sids)

    def make_decoder_only_data(src, tgt):
        label, token_type_ids = _tokenize_decoder_only(
            src,
            tgt,
            seqlen=seqlen,
        )
        assert len(label) <= seqlen, (len(label), seqlen)
        ids, label = label[:-1], label[1:]
        token_type_ids = token_type_ids[:-1]
        mask = (ids != tokenizer.ignored_index).astype("int32")
        ret = dict(
            token_type_ids=token_type_ids,
            attention_mask=mask,
            input_ids=ids,
            labels=label,
        )
        return ret

    def _gen():
        nonlocal filelist
        # assert len(filelist) >= dp_size, f'dp_size:{dp_size} < #file:{len(filelist)}'
        worker_id, num_worker = 0, 1
        if len(filelist) > dp_size:
            filelist = filelist[dp_rank::dp_size]
            logger.info(
                f"sharding filelist, after shard:{filelist} dp={dp_rank}/{dp_size}"
            )
        else:
            logger.info(f"not sharding filelist:{filelist}")
            worker_info = paddle.io.get_worker_info()
            if worker_info is not None:
                num_worker = worker_info.num_workers
                worker_id = worker_info.id
                logger.info(
                    f"using reader multiprocess={worker_id}/{num_worker},dp={dp_rank}/{dp_size}"
                )

        def file_reader(file):
            with open(file, "rb") as inf:
                for cnt, line in enumerate(inf):
                    try:
                        line = line.decode(task_processor.encoding)
                        yield cnt, line
                    except UnicodeDecodeError:
                        logger.info(
                            f"unicode decode error, task={task_processor}, #line={cnt}, file={file}"
                        )
                        continue

        def gzip_file_reader(file):
            with gzip.open(file, "rb") as inf:
                for cnt, line in enumerate(inf):
                    try:
                        line = line.decode(task_processor.encoding)
                        yield cnt, line
                    except UnicodeDecodeError:
                        logger.info(
                            f"gzip unicode decode error, task={task_processor}, #line={cnt}, file={file}"
                        )
                        continue

        epoch, buffer = 0, []
        while 1:
            for file in filelist:
                reader = gzip_file_reader if file.endswith("gz") else file_reader
                for cnt, line in reader(file):
                    if cnt % num_worker != worker_id:
                        continue
                    try:
                        ret = task_processor(line)
                    except Exception as e:
                        logger.exception(e)
                        logger.warn(
                            f"data process failed, continue, task={task_processor}, #line={cnt}, file={file} err={e}"
                        )
                        continue
                    if not ret:
                        continue
                    src, tgt = ret
                    if is_decoder_only:
                        ret = make_decoder_only_data(src, tgt)
                    else:
                        ret = make_encoder_decoder_data(src, tgt)
                    if ret is None:
                        continue
                    if not no_pad:
                        yield ret
                    else:
                        buffer.append(ret)
                        if sum([len(b["input_ids"]) for b in buffer]) > seqlen:
                            ret = {}
                            for k in buffer[0].keys():
                                ret[k] = paddle.concat([b[k] for b in buffer], 0)[
                                    :seqlen
                                ]
                            yield ret
                            buffer = []
            epoch += 1
            logger.info(f"task={task_processor}, epoch={epoch}.")

    return _gen


mix_of_processor = {
    "json": JsonProcessor,
    "tsv": TsvProcessor,
}


def make_weighted_reader(
    multi_task,
    tokenizer,
    rng,
    seqlen,
    dp_size,
    dp_rank,
    is_decoder_only=False,
    no_pad=False,
):
    def _gen():
        processors = [
            mix_of_processor[task_spec["processor_type"]](
                tokenizer=tokenizer, **task_spec
            )
            for task, task_spec in multi_task.items()
        ]
        gens = [
            make_reader(
                glob(task_spec["paths"]),
                tokenizer,
                processor,
                rng,
                seqlen,
                dp_size=dp_size,
                dp_rank=dp_rank,
                is_decoder_only=is_decoder_only,
                no_pad=no_pad,
            )
            for (task, task_spec), processor in zip(multi_task.items(), processors)
        ]
        gens = [g() for g in gens]
        weights = [task_spec["weight"] for task, task_spec in multi_task.items()]
        weights = [w / sum(weights) for w in weights]
        cum_weights = list(itertools.accumulate(weights))
        for task, g, w in zip(multi_task.keys(), gens, weights):
            ex = next(g)
            worker_info = paddle.io.get_worker_info()
            if (
                worker_info is not None and worker_info.id != 0
            ):  # log once in multi processor
                continue
            debug_src = tokenizer.decode(ex["input_ids"])
            debug_tgt = tokenizer.decode(ex["labels"])
            debuginfo = (
                f'id>>{ex["input_ids"]}\n input>> {debug_src}\nlabel>> {debug_tgt}'
            )
            logger.info(
                f"Task Example:{task}, is_decoder_only={is_decoder_only}\n Example={debuginfo}, \nweights={w}"
            )

        try:
            while 1:
                g = rng.choices(gens, cum_weights=cum_weights, k=1)[0]
                data = next(g)
                yield data
        except StopIteration:
            pass

    return _gen


if __name__ == "__main__":
    import argparse
    from transformers import BertTokenizer
    from transformers import DebertaV2Tokenizer

    log = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    stream_hdl = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:    %(message)s"
    )
    stream_hdl.setFormatter(formatter)
    logger.handlers = [stream_hdl]
    parser = argparse.ArgumentParser(description="main")
    # parser.add_argument("-d", "--data_dir", type=str,  nargs='+')
    parser.add_argument("--tok", required=True, type=str)
    parser.add_argument("--task-spec", required=True, type=str)
    parser.add_argument("--seqlen", type=int, default=512)
    args = parser.parse_args()
    if "deberta" in args.tok:
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.tok)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.tok)
    tokenizer.ignored_index = 0
    rng = random.Random(0)
    multi_task = json.loads(open(args.task_spec).read())
    for data in make_weighted_reader(multi_task, tokenizer, rng, args.seqlen, 1, 0)():
        logger.info("*****")
        logger.info({f"{k}:{v.shape}" for k, v in data.items()})
        id = data["input_ids"]
        label_id = data["labels"]
        src = tokenizer.decode(id)
        label = tokenizer.decode(label_id)
        logger.info(f"id>>{id}\n")
        print("-------")
        print("**src**=" + src.replace("[unused88]", "\n"))
        print("**tgt**=" + label.replace("[unused88]", "\n"))
        print("-------")
