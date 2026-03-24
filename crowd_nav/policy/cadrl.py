import torch
import torch.nn as nn
import numpy as np
import itertools
import logging
import os
from collections.abc import Mapping
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        value = self.value_network(state)
        return value


class CADRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'CADRL'
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

        self.inference_backend = 'torch'
        self._ov_runtime = None
        self._ov_core = None
        self._ov_model_path = None
        self._ov_export_path = None
        self._ov_device = 'CPU'
        self._ov_performance_hint = 'LATENCY'
        self._ov_num_threads = 0
        self._ov_cache_dir = None
        self._ov_compiled_cache = dict()
        self._ov_training_warning_emitted = False

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('cadrl', 'mlp_dims').split(', ')]
        self.model = ValueNetwork(self.joint_state_dim, mlp_dims)
        self.multiagent_training = config.getboolean('cadrl', 'multiagent_training')
        self.configure_inference_backend(config)
        logging.info('Policy: CADRL without occupancy map')

    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def configure_inference_backend(self, config):
        if not config.has_section('inference'):
            self.disable_openvino()
            return
        backend = config.get('inference', 'backend', fallback='torch').strip().lower()
        if backend in ['', 'torch', 'pytorch', 'pt']:
            self.disable_openvino()
            return
        if backend != 'openvino':
            raise ValueError("Unsupported inference backend '{}'. Use 'torch' or 'openvino'.".format(backend))

        self.enable_openvino(
            model_path=config.get('inference', 'openvino_model_path', fallback='').strip() or None,
            device_name=config.get('inference', 'openvino_device', fallback='CPU').strip() or 'CPU',
            performance_hint=config.get('inference', 'openvino_performance_hint', fallback='LATENCY').strip() or 'LATENCY',
            num_threads=config.getint('inference', 'openvino_num_threads', fallback=0),
            cache_dir=config.get('inference', 'openvino_cache_dir', fallback='').strip() or None,
            export_path=config.get('inference', 'openvino_export_path', fallback='').strip() or None,
        )

    def enable_openvino(
        self,
        model_path=None,
        device_name='CPU',
        performance_hint='LATENCY',
        num_threads=0,
        cache_dir=None,
        export_path=None,
    ):
        self._ensure_openvino_runtime()
        if model_path is not None and not os.path.exists(model_path):
            raise FileNotFoundError("OpenVINO model file '{}' does not exist.".format(model_path))
        self.inference_backend = 'openvino'
        self._ov_model_path = model_path
        self._ov_export_path = export_path
        self._ov_device = device_name
        self._ov_performance_hint = performance_hint
        self._ov_num_threads = max(0, int(num_threads))
        self._ov_cache_dir = cache_dir
        self._ov_compiled_cache.clear()
        self._ov_core = None
        if hasattr(self.model, 'attention_weights'):
            self.model.attention_weights = None
        logging.info('OpenVINO inference enabled on device %s', self._ov_device)

    def disable_openvino(self):
        self.inference_backend = 'torch'
        self._ov_compiled_cache.clear()

    def export_openvino_model(self, output_xml_path, input_shape):
        ov, _ = self._ensure_openvino_runtime()
        output_xml_path = self._normalize_openvino_xml_path(output_xml_path)
        output_dir = os.path.dirname(output_xml_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        dummy_input = torch.randn(*input_shape, dtype=torch.float32)
        model_device = self._model_device()
        self.model.to(torch.device('cpu'))
        self.model.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(self.model, example_input=dummy_input)
        ov.save_model(ov_model, output_xml_path)
        self.model.to(model_device)
        self._ov_model_path = output_xml_path
        logging.info('Exported OpenVINO IR model: %s', output_xml_path)
        return output_xml_path

    def _model_device(self):
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return torch.device('cpu')
        return first_param.device

    def _normalize_openvino_xml_path(self, output_path):
        if output_path is None:
            return None
        if output_path.endswith('.xml'):
            return output_path
        return os.path.join(output_path, '{}.xml'.format(self.name.lower()))

    def _ensure_openvino_runtime(self):
        if self._ov_runtime is None:
            try:
                import openvino as ov
            except ImportError as exc:
                raise RuntimeError(
                    'OpenVINO backend requested but openvino is not installed. Install with `pip install openvino`.'
                ) from exc
            self._ov_runtime = ov

        if self._ov_core is None:
            self._ov_core = self._ov_runtime.Core()
            if self._ov_cache_dir:
                os.makedirs(self._ov_cache_dir, exist_ok=True)
                try:
                    self._ov_core.set_property({'CACHE_DIR': self._ov_cache_dir})
                except Exception as exc:
                    logging.warning('Failed to set OpenVINO CACHE_DIR=%s (%s)', self._ov_cache_dir, exc)

        return self._ov_runtime, self._ov_core

    @staticmethod
    def _ov_port_name(port):
        if hasattr(port, 'any_name'):
            return port.any_name
        if hasattr(port, 'get_any_name'):
            return port.get_any_name()
        return None

    def _compile_openvino_model(self, input_tensor):
        _, core = self._ensure_openvino_runtime()
        shape_key = tuple(int(x) for x in input_tensor.shape)
        if shape_key in self._ov_compiled_cache:
            return self._ov_compiled_cache[shape_key]

        if self._ov_model_path:
            ov_model = core.read_model(self._ov_model_path)
            input_port = ov_model.inputs[0]
            input_name = self._ov_port_name(input_port)
            try:
                if input_name:
                    ov_model.reshape({input_name: list(shape_key)})
                else:
                    ov_model.reshape({input_port: list(shape_key)})
            except Exception as exc:
                logging.debug('OpenVINO reshape skipped for shape %s (%s)', shape_key, exc)
        else:
            ov_model = self._convert_torch_model_to_openvino(input_tensor)
            if self._ov_export_path:
                output_xml_path = self._normalize_openvino_xml_path(self._ov_export_path)
                output_dir = os.path.dirname(output_xml_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                self._ov_runtime.save_model(ov_model, output_xml_path)
                self._ov_model_path = output_xml_path
                logging.info('Saved converted OpenVINO model to %s', output_xml_path)

        compile_config = {}
        if self._ov_performance_hint:
            compile_config['PERFORMANCE_HINT'] = self._ov_performance_hint
        if self._ov_num_threads > 0:
            compile_config['INFERENCE_NUM_THREADS'] = str(self._ov_num_threads)

        if compile_config:
            compiled_model = core.compile_model(ov_model, self._ov_device, compile_config)
        else:
            compiled_model = core.compile_model(ov_model, self._ov_device)
        self._ov_compiled_cache[shape_key] = compiled_model
        return compiled_model

    def _convert_torch_model_to_openvino(self, input_tensor):
        ov, _ = self._ensure_openvino_runtime()
        model_device = self._model_device()
        self.model.to(torch.device('cpu'))
        self.model.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(
                self.model,
                example_input=input_tensor.detach().to(torch.device('cpu')),
            )
        self.model.to(model_device)
        return ov_model

    @staticmethod
    def _to_numpy_output(value):
        if isinstance(value, np.ndarray):
            if value.dtype == object and value.size == 1:
                return CADRL._to_numpy_output(value.item())
            return value
        if hasattr(value, 'data'):
            return CADRL._to_numpy_output(value.data)
        return np.asarray(value)

    def _openvino_infer_numpy(self, input_tensor):
        input_numpy = input_tensor.detach().to(torch.device('cpu')).numpy().astype(np.float32, copy=False)
        compiled_model = self._compile_openvino_model(input_tensor)
        infer_request = compiled_model.create_infer_request()
        input_name = self._ov_port_name(compiled_model.input(0))
        try:
            if input_name:
                outputs = infer_request.infer({input_name: input_numpy})
            else:
                outputs = infer_request.infer([input_numpy])
        except Exception:
            outputs = infer_request.infer([input_numpy])

        if isinstance(outputs, Mapping):
            output_port = compiled_model.output(0)
            output_array = outputs.get(output_port)
            if output_array is None:
                output_array = next(iter(outputs.values()))
            return CADRL._to_numpy_output(output_array)
        return CADRL._to_numpy_output(outputs)

    def infer_value_network_numpy(self, input_tensor):
        if self.inference_backend == 'openvino' and self.phase != 'train':
            return self._openvino_infer_numpy(input_tensor)

        if self.inference_backend == 'openvino' and self.phase == 'train' and not self._ov_training_warning_emitted:
            logging.warning('OpenVINO backend is disabled in train phase. Falling back to PyTorch for sampling.')
            self._ov_training_warning_emitted = True

        with torch.no_grad():
            outputs = self.model(input_tensor)
        return outputs.detach().cpu().numpy()

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                       state.gx, state.gy, state.v_pref, state.theta)
            else:
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
                next_px = state.px + next_vx * self.time_step
                next_py = state.py + next_vy * self.time_step
                next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                       state.v_pref, next_theta)
        else:
            raise ValueError('Type error')

        return next_state

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_min_value = float('-inf')
            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                ob, reward, done, info = self.env.onestep_lookahead(action)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                              for next_human_state in ob], dim=0)
                # VALUE UPDATE
                outputs = self.infer_value_network_numpy(self.rotate(batch_next_states))
                min_output = float(np.min(outputs))
                min_value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * min_output
                self.action_values.append(min_value)
                if min_value > max_min_value:
                    max_min_value = min_value
                    max_action = action

        if self.phase == 'train':
            self.last_state = self.transform(state)

        return max_action

    def transform(self, state):
        """
        Take the state passed from agent and transform it to tensor for batch training

        :param state:
        :return: tensor of shape (len(state), )
        """
        assert len(state.human_states) == 1
        state = torch.Tensor(state.self_state + state.human_states[0]).to(self.device)
        state = self.rotate(state.unsqueeze(0)).squeeze(dim=0)
        return state

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state
