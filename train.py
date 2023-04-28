import argparse
import os
import torch
import shutil
import pickle
import re
import collections
import json

from scale_configs import get_scale_config, available_scales

from pathlib import Path
from cloudpathlib import CloudPath

from training.main import main
from training.distributed import world_info_from_env


def prepare_filename(filename):
    filename = str(filename)
    if filename.startswith('s3://'):
        return f'pipe:aws s3 cp {filename} -'
    return filename


def split_filename(pattern, filename):
    filename = str(filename)
    pattern_match = pattern.search(filename)
    pos = pattern_match.start()
    return filename[:pos], filename[pos:]


def get_input_shards(data_dir, weights):
    # Handle multiple directories
    if '::' in str(data_dir):
        split_data_dir = str(data_dir).split('::')
        data_dirs = [path_or_cloudpath(subdir) for subdir in split_data_dir]
        if weights is None:
            split_weights = [None for _ in split_data_dir]
        else:
            split_weights = weights.split('::')
            assert len(split_weights) == len(split_data_dir)
            
        input_strs_and_weights = [get_input_shards(subdir, weight) for (subdir, weight) in zip(data_dirs, split_weights)]

        input_strs, input_weights = zip(*input_strs_and_weights)
        input_strs = '::'.join(input_strs)
        if weights is not None:
            weights = '::'.join(input_weights)
        return input_strs, weights 

    # Handle raw shards
    if data_dir.suffix == '.tar':
        return prepare_filename(data_dir), weights

    # Handle folders
    files_or_subdirs = list(data_dir.iterdir())
    data_str_components = []
    prefix_map = collections.defaultdict(list)
    pattern = re.compile('\d+$')  # Sequence of digits at the end of the string
    count_tars = 0
    for file_or_subdir in files_or_subdirs:
        if file_or_subdir.suffix == '.tar':
            shard = file_or_subdir.with_suffix('')
            prefix, suffix = split_filename(pattern, shard)
            prefix_map[prefix].append(suffix)
            count_tars += 1
        elif file_or_subdir.is_dir():
            # If the folder is generated by the resharder, the metadata file contains how many shards there are.
            metadata_file = file_or_subdir / 'meta.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                shard_count = metadata['output_shard_count']
                shard_format = metadata['output_shard_format']
                first_shard = shard_format.format(0).replace(".tar", "")
                last_shard = shard_format.format(shard_count-1).replace(".tar", "")
                filename = f'{{{first_shard}..{last_shard}}}.tar'
                subfolder_str = prepare_filename(file_or_subdir / filename)
                data_str_components.append(subfolder_str)
            else:
                sub_data_strs, _ = get_input_shards(file_or_subdir, weights)
                data_str_components.extend(sub_data_strs.split('::'))
        
    for prefix in sorted(list(prefix_map.keys())):
        last_tar = max([int(suffix) for suffix in prefix_map[prefix]])
        number_of_zeros = len(prefix_map[prefix][0])
        filename = f'{{{0:0{number_of_zeros}d}..{last_tar:0{number_of_zeros}d}}}.tar'
        filename = prepare_filename(prefix + filename)
        data_str_components.append(filename)
    data_str = '::'.join(data_str_components)
    if weights is not None:
        weights = '::'.join([weights for _ in data_str_components])
    return data_str, weights


def path_or_cloudpath(s):
    if re.match(r"^\w+://", s):
        return CloudPath(s)
    return Path(s)


def save_training_artifacts(args, config, checkpoint):
    training_artifacts = {
        'scale': args.scale,
        'checkpoint': checkpoint,
        'scale_config': config,
        'data_dir': args.data_dir
    }
    artifacts_fname = checkpoint.parent.parent / 'info.pkl'
    pickle.dump(training_artifacts, open(artifacts_fname, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--scale',
        type=str,
        required=True,
        choices=available_scales(),
        help='Competition scale.'
    )
    parser.add_argument(
        '--data_dir',
        type=path_or_cloudpath,
        required=True,
        help='Path to directory where the data is stored. Multiple paths can be used, separated by "::".'
    )
    parser.add_argument(
        "--data_weights",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, which weight to use for sampling the different data sources. "
            "Similar to --data-dir, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        '--output_dir',
        type=path_or_cloudpath,
        required=True,
        help='Path to directory where outputs will be stored.'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Name of the experiment for logging.'
    )
    parser.add_argument(
        '--use_cached_shards',
        help='If true, re-use the re-sharded data if possible.',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--wandb_project_name',
        type=str,
        default='datanet',
        help='Name of the project if logging with wandb.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of workers for open_clip.'
    )
    parser.add_argument(
        '--precision',
        type=str, choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        '--num_checkpoints',
        type=int,
        default=5,
        help="Number of times we save checkpoints during training."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Random seed."
    )
    parser.add_argument(
        "--dataset_resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--report_to_wandb",
        default=False,
        action="store_true",
        help="If True, report to wandb."
    )
    parser.add_argument(
        "--accum_freq",
        type=int,
        default=1,
        help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--resume",
        default='latest',
        type=str,
        help="Path to checkpoint to resume from (default: latest checkpoint in the training directory).",
    )
    parser.add_argument(
        "--imagenet_val",
        type=str,
        default=None,
        help="Optional path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--blur_field",
        type=str,
        default=None,
        help="Name of the field in the webdataset json files with bounding boxes to blur."
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=None
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=0
    )

    args = parser.parse_args()
    data_dir = args.data_dir

    _, rank, world_size = world_info_from_env()
    if rank == 0:
        print('Running training on scale', args.scale)
        print(f'World size is {world_size}.')
    config = get_scale_config(args.scale)
    learning_rate = config['learning_rate']
    global_batch_size = config['batch_size']
    warmup = config['warmup']
    model = config['model']
    beta2 = config['beta2']
    train_num_samples = config['train_num_samples']
    train_data, weights = get_input_shards(data_dir, args.data_weights)

    exp_name = args.exp_name if args.exp_name else f'{args.scale}_scale'

    log_dir = args.output_dir

    per_gpu_batch_size = global_batch_size // (world_size * args.accum_freq)

    main_args = [
        '--save-frequency', f'{args.save_frequency}',
        '--ddp-static-graph',
        '--local-loss',
        '--gather-with-grad',
        '--grad-checkpointing',
        '--train-data', f'{train_data}',
        '--train-num-samples', f'{train_num_samples // args.num_checkpoints}',
        '--warmup', f'{warmup}',
        '--dataset-type', 'webdataset',
        '--precision', f'{args.precision}',
        '--workers', f'{args.workers}',
        '--model', f'{model}',
        '--batch-size', f'{per_gpu_batch_size}',
        '--epochs', f'{args.num_checkpoints}',
        '--lr', f'{learning_rate}',
        '--logs', f'{log_dir}',
        '--name', f'{exp_name}',
        '--seed', f'{args.seed}',
        '--accum-freq', f'{args.accum_freq}',
        '--log-every-n-steps', f'{args.log_every_n_steps}',
        '--save-most-recent',
        '--resume', f'{args.resume}'
    ]
    if args.dataset_resampled:
        main_args.append('--dataset-resampled')
    if args.report_to_wandb:
        main_args.extend(['--report-to', 'wandb', '--wandb-project-name', f'{args.wandb_project_name}'])
    if args.imagenet_val is not None:
        main_args.extend(['--imagenet-val', args.imagenet_val])
    if args.blur_field is not None:
        main_args.extend(['--blur-field', args.blur_field])
    if beta2 is not None:
        main_args.extend(['--beta2', f'{beta2}'])
    if weights is not None:
        main_args.extend(['--train-data-upsampling-factors', weights])
    if args.grad_clip_norm is not None:
        main_args.extend(['--grad-clip-norm', f'{args.grad_clip_norm}'])

    success = main(main_args)
    
    if rank == 0:
        if success == -1:
            print('Error running training. Exiting.')
        
        final_checkpoint = log_dir / exp_name / 'checkpoints' / f'epoch_latest.pt'
        assert final_checkpoint.exists(), f'Did not find the checkpoint at {final_checkpoint}'
        save_training_artifacts(args, config, final_checkpoint)

        print('Done training.')
