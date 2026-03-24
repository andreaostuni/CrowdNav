import argparse
import configparser
import logging
import os

import torch

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.multi_human_rl import MultiHumanRL


def extract_state_dict(checkpoint, state_dict_key=''):
    if isinstance(checkpoint, dict):
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
        keys_to_try = [state_dict_key] if state_dict_key else []
        keys_to_try += ['state_dict', 'policy_state_dict', 'model_state_dict']
        for key in keys_to_try:
            if key and key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return None


def build_export_input_shape(policy, num_humans):
    if isinstance(policy, MultiHumanRL):
        return (1, num_humans, policy.input_dim())
    return (num_humans, policy.joint_state_dim)


def parse_args():
    parser = argparse.ArgumentParser('Export CrowdNav policy to OpenVINO IR')
    parser.add_argument('--policy', type=str, default='sarl')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--state_dict_key', type=str, default='')
    parser.add_argument('--num_humans', type=int, default=5)
    parser.add_argument('--output', type=str, required=True,
                        help='Target .xml file path, or output directory if extension is omitted')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    if args.policy not in policy_factory:
        raise ValueError("Unknown policy '{}'. Available: {}".format(args.policy, sorted(policy_factory.keys())))

    if not os.path.exists(args.policy_config):
        raise FileNotFoundError("Policy config '{}' does not exist".format(args.policy_config))
    if not os.path.exists(args.weights):
        raise FileNotFoundError("Weights file '{}' does not exist".format(args.weights))

    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)

    if not policy.trainable:
        raise ValueError("Policy '{}' is not trainable and cannot be exported".format(args.policy))

    checkpoint = torch.load(args.weights, map_location='cpu')
    state_dict = extract_state_dict(checkpoint, args.state_dict_key)
    if state_dict is None:
        raise ValueError(
            'Unable to extract a state dict from checkpoint. Use --state_dict_key if your checkpoint nests it.'
        )

    policy.get_model().load_state_dict(state_dict)
    policy.set_device(torch.device('cpu'))
    policy.set_phase('test')

    if not hasattr(policy, 'export_openvino_model'):
        raise TypeError("Policy '{}' does not expose export_openvino_model()".format(args.policy))

    input_shape = build_export_input_shape(policy, args.num_humans)
    output_xml = policy.export_openvino_model(args.output, input_shape)

    logging.info('Export complete')
    logging.info('Policy: %s', args.policy)
    logging.info('Input shape used for export: %s', input_shape)
    logging.info('OpenVINO IR XML: %s', output_xml)


if __name__ == '__main__':
    main()
