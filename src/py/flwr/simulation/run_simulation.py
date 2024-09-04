# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
# ==============================================================================
"""Flower Simulation."""

import argparse
import asyncio
import json
import logging
import threading
import traceback
from logging import DEBUG, ERROR, INFO, WARNING
from time import sleep
from typing import Optional

from flwr.client import ClientApp
from flwr.common import EventType, event, log
from flwr.common.logger import set_logger_propagation, update_console_handler
from flwr.common.typing import Run
from flwr.server.driver import Driver, InMemoryDriver
from flwr.server.run_serverapp import run
from flwr.server.server_app import ServerApp
from flwr.server.superlink.fleet import vce
from flwr.server.superlink.fleet.vce.backend.backend import BackendConfig
from flwr.server.superlink.state import StateFactory
from flwr.simulation.ray_transport.utils import (
    enable_tf_gpu_growth as enable_gpu_growth,
)


# Entry point from CLI
def run_simulation_from_cli() -> None:
    """Run Simulation Engine from the CLI."""
    args = _parse_args_run_simulation().parse_args()

    # Load JSON config
    backend_config_dict = json.loads(args.backend_config)

    _run_simulation(
        server_app_attr=args.server_app,
        client_app_attr=args.client_app,
        num_supernodes=args.num_supernodes,
        backend_name=args.backend,
        backend_config=backend_config_dict,
        app_dir=args.app_dir,
        run_id=args.run_id,
        enable_tf_gpu_growth=args.enable_tf_gpu_growth,
        verbose_logging=args.verbose,
    )


# Entry point from Python session (script or notebook)
# pylint: disable=too-many-arguments
def run_simulation(
    server_app: ServerApp,
    client_app: ClientApp,
    num_supernodes: int,
    backend_name: str = "ray",
    backend_config: Optional[BackendConfig] = None,
    enable_tf_gpu_growth: bool = False,
    verbose_logging: bool = False,
) -> None:
    r"""Run a Flower App using the Simulation Engine.

    Parameters
    ----------
    server_app : ServerApp
        The `ServerApp` to be executed. It will send messages to different `ClientApp`
        instances running on different (virtual) SuperNodes.

    client_app : ClientApp
        The `ClientApp` to be executed by each of the SuperNodes. It will receive
        messages sent by the `ServerApp`.

    num_supernodes : int
        Number of nodes that run a ClientApp. They can be sampled by a
        Driver in the ServerApp and receive a Message describing what the ClientApp
        should perform.

    backend_name : str (default: ray)
        A simulation backend that runs `ClientApp`s.

    backend_config : Optional[BackendConfig]
        'A dictionary to configure a backend. Separate dictionaries to configure
        different elements of backend. Supported top-level keys are `init_args`
        for values parsed to initialisation of backend, `client_resources`
        to define the resources for clients, and `actor` to define the actor
        parameters. Values supported in <value> are those included by
        `flwr.common.typing.ConfigsRecordValues`.

    enable_tf_gpu_growth : bool (default: False)
        A boolean to indicate whether to enable GPU growth on the main thread. This is
        desirable if you make use of a TensorFlow model on your `ServerApp` while
        having your `ClientApp` running on the same GPU. Without enabling this, you
        might encounter an out-of-memory error because TensorFlow, by default, allocates
        all GPU memory. Read more about how `tf.config.experimental.set_memory_growth()`
        works in the TensorFlow documentation: https://www.tensorflow.org/api/stable.

    verbose_logging : bool (default: False)
        When disabled, only INFO, WARNING and ERROR log messages will be shown. If
        enabled, DEBUG-level logs will be displayed.
    """
    _run_simulation(
        num_supernodes=num_supernodes,
        client_app=client_app,
        server_app=server_app,
        backend_name=backend_name,
        backend_config=backend_config,
        enable_tf_gpu_growth=enable_tf_gpu_growth,
        verbose_logging=verbose_logging,
    )


# pylint: disable=too-many-arguments
def run_serverapp_th(
    server_app_attr: Optional[str],
    server_app: Optional[ServerApp],
    driver: Driver,
    app_dir: str,
    f_stop: asyncio.Event,
    enable_tf_gpu_growth: bool,
    delay_launch: int = 3,
) -> threading.Thread:
    """Run SeverApp in a thread."""

    def server_th_with_start_checks(  # type: ignore
        tf_gpu_growth: bool, stop_event: asyncio.Event, **kwargs
    ) -> None:
        """Run SeverApp, after check if GPU memory growth has to be set.

        Upon exception, trigger stop event for Simulation Engine.
        """
        try:
            if tf_gpu_growth:
                log(INFO, "Enabling GPU growth for Tensorflow on the main thread.")
                enable_gpu_growth()

            # Run ServerApp
            run(**kwargs)
        except Exception as ex:  # pylint: disable=broad-exception-caught
            log(ERROR, "ServerApp thread raised an exception: %s", ex)
            log(ERROR, traceback.format_exc())
        finally:
            log(DEBUG, "ServerApp finished running.")
            # Upon completion, trigger stop event if one was passed
            if stop_event is not None:
                stop_event.set()
                log(DEBUG, "Triggered stop event for Simulation Engine.")

    serverapp_th = threading.Thread(
        target=server_th_with_start_checks,
        args=(enable_tf_gpu_growth, f_stop),
        kwargs={
            "server_app_attr": server_app_attr,
            "loaded_server_app": server_app,
            "driver": driver,
            "server_app_dir": app_dir,
        },
    )
    sleep(delay_launch)
    serverapp_th.start()
    return serverapp_th


def _override_run_id(state: StateFactory, run_id_to_replace: int, run_id: int) -> None:
    """Override the run_id of an existing Run."""
    log(DEBUG, "Pre-registering run with id %s", run_id)
    # Remove run
    run_info: Run = state.state().run_ids.pop(run_id_to_replace)  # type: ignore
    # Update with new run_id and insert back in state
    run_info.run_id = run_id
    state.state().run_ids[run_id] = run_info  # type: ignore


# pylint: disable=too-many-locals
def _main_loop(
    num_supernodes: int,
    backend_name: str,
    backend_config_stream: str,
    app_dir: str,
    enable_tf_gpu_growth: bool,
    run_id: Optional[int] = None,
    client_app: Optional[ClientApp] = None,
    client_app_attr: Optional[str] = None,
    server_app: Optional[ServerApp] = None,
    server_app_attr: Optional[str] = None,
) -> None:
    """Launch SuperLink with Simulation Engine, then ServerApp on a separate thread.

    Everything runs on the main thread or a separate one, depending on whether the main
    thread already contains a running Asyncio event loop. This is the case if running
    the Simulation Engine on a Jupyter/Colab notebook.
    """
    # Initialize StateFactory
    state_factory = StateFactory(":flwr-in-memory-state:")

    f_stop = asyncio.Event()
    serverapp_th = None
    try:
        # Create run (with empty fab_id and fab_version)
        run_id_ = state_factory.state().create_run("", "")

        if run_id:
            _override_run_id(state_factory, run_id_to_replace=run_id_, run_id=run_id)
            run_id_ = run_id

        # Initialize Driver
        driver = InMemoryDriver(run_id=run_id_, state_factory=state_factory)

        # Get and run ServerApp thread
        serverapp_th = run_serverapp_th(
            server_app_attr=server_app_attr,
            server_app=server_app,
            driver=driver,
            app_dir=app_dir,
            f_stop=f_stop,
            enable_tf_gpu_growth=enable_tf_gpu_growth,
        )

        # SuperLink with Simulation Engine
        event(EventType.RUN_SUPERLINK_ENTER)
        vce.start_vce(
            num_supernodes=num_supernodes,
            client_app_attr=client_app_attr,
            client_app=client_app,
            backend_name=backend_name,
            backend_config_json_stream=backend_config_stream,
            app_dir=app_dir,
            state_factory=state_factory,
            f_stop=f_stop,
        )

    except Exception as ex:
        log(ERROR, "An exception occurred !! %s", ex)
        log(ERROR, traceback.format_exc())
        raise RuntimeError("An error was encountered. Ending simulation.") from ex

    finally:
        # Trigger stop event
        f_stop.set()

        event(EventType.RUN_SUPERLINK_LEAVE)
        if serverapp_th:
            serverapp_th.join()

    log(DEBUG, "Stopping Simulation Engine now.")


# pylint: disable=too-many-arguments,too-many-locals
def _run_simulation(
    num_supernodes: int,
    client_app: Optional[ClientApp] = None,
    server_app: Optional[ServerApp] = None,
    backend_name: str = "ray",
    backend_config: Optional[BackendConfig] = None,
    client_app_attr: Optional[str] = None,
    server_app_attr: Optional[str] = None,
    app_dir: str = "",
    run_id: Optional[int] = None,
    enable_tf_gpu_growth: bool = False,
    verbose_logging: bool = False,
) -> None:
    r"""Launch the Simulation Engine.

    Parameters
    ----------
    num_supernodes : int
        Number of nodes that run a ClientApp. They can be sampled by a
        Driver in the ServerApp and receive a Message describing what the ClientApp
        should perform.

    client_app : Optional[ClientApp]
        The `ClientApp` to be executed by each of the `SuperNodes`. It will receive
        messages sent by the `ServerApp`.

    server_app : Optional[ServerApp]
        The `ServerApp` to be executed.

    backend_name : str (default: ray)
        A simulation backend that runs `ClientApp`s.

    backend_config : Optional[BackendConfig]
        'A dictionary to configure a backend. Separate dictionaries to configure
        different elements of backend. Supported top-level keys are `init_args`
        for values parsed to initialisation of backend, `client_resources`
        to define the resources for clients, and `actor` to define the actor
        parameters. Values supported in <value> are those included by
        `flwr.common.typing.ConfigsRecordValues`.

    client_app_attr : str
        A path to a `ClientApp` module to be loaded: For example: `client:app` or
        `project.package.module:wrapper.app`."

    server_app_attr : str
        A path to a `ServerApp` module to be loaded: For example: `server:app` or
        `project.package.module:wrapper.app`."

    app_dir : str
        Add specified directory to the PYTHONPATH and load `ClientApp` from there.
        (Default: current working directory.)

    run_id : Optional[int]
        An integer specifying the ID of the run started when running this function.

    enable_tf_gpu_growth : bool (default: False)
        A boolean to indicate whether to enable GPU growth on the main thread. This is
        desirable if you make use of a TensorFlow model on your `ServerApp` while
        having your `ClientApp` running on the same GPU. Without enabling this, you
        might encounter an out-of-memory error because TensorFlow by default allocates
        all GPU memory. Read mor about how `tf.config.experimental.set_memory_growth()`
        works in the TensorFlow documentation: https://www.tensorflow.org/api/stable.

    verbose_logging : bool (default: False)
        When disabled, only INFO, WARNING and ERROR log messages will be shown. If
        enabled, DEBUG-level logs will be displayed.
    """
    if backend_config is None:
        backend_config = {}

    if "init_args" not in backend_config:
        backend_config["init_args"] = {}

    # Set logging level
    logger = logging.getLogger("flwr")
    if verbose_logging:
        update_console_handler(level=DEBUG, timestamps=True, colored=True)
    else:
        backend_config["init_args"]["logging_level"] = WARNING
        backend_config["init_args"]["log_to_driver"] = True

    if enable_tf_gpu_growth:
        # Check that Backend config has also enabled using GPU growth
        use_tf = backend_config.get("actor", {}).get("tensorflow", False)
        if not use_tf:
            log(WARNING, "Enabling GPU growth for your backend.")
            backend_config["actor"]["tensorflow"] = True

    # Convert config to original JSON-stream format
    backend_config_stream = json.dumps(backend_config)

    simulation_engine_th = None
    args = (
        num_supernodes,
        backend_name,
        backend_config_stream,
        app_dir,
        enable_tf_gpu_growth,
        run_id,
        client_app,
        client_app_attr,
        server_app,
        server_app_attr,
    )
    # Detect if there is an Asyncio event loop already running.
    # If yes, run everything on a separate thread. In environments
    # like Jupyter/Colab notebooks, there is an event loop present.
    run_in_thread = False
    try:
        _ = (
            asyncio.get_running_loop()
        )  # Raises RuntimeError if no event loop is present
        log(DEBUG, "Asyncio event loop already running.")

        run_in_thread = True

    except RuntimeError:
        log(DEBUG, "No asyncio event loop running")

    finally:
        if run_in_thread:
            # Set logger propagation to False to prevent duplicated log output in Colab.
            logger = set_logger_propagation(logger, False)
            log(DEBUG, "Starting Simulation Engine on a new thread.")
            simulation_engine_th = threading.Thread(target=_main_loop, args=args)
            simulation_engine_th.start()
            simulation_engine_th.join()
        else:
            log(DEBUG, "Starting Simulation Engine on the main thread.")
            _main_loop(*args)


def _parse_args_run_simulation() -> argparse.ArgumentParser:
    """Parse flower-simulation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Flower simulation",
    )
    parser.add_argument(
        "--server-app",
        required=True,
        help="For example: `server:app` or `project.package.module:wrapper.app`",
    )
    parser.add_argument(
        "--client-app",
        required=True,
        help="For example: `client:app` or `project.package.module:wrapper.app`",
    )
    parser.add_argument(
        "--num-supernodes",
        type=int,
        required=True,
        help="Number of simulated SuperNodes.",
    )
    parser.add_argument(
        "--backend",
        default="ray",
        type=str,
        help="Simulation backend that executes the ClientApp.",
    )
    parser.add_argument(
        "--backend-config",
        type=str,
        default='{"client_resources": {"num_cpus":2, "num_gpus":0.0},'
        '"actor": {"tensorflow": 0}}',
        help='A JSON formatted stream, e.g \'{"<keyA>":<value>, "<keyB>":<value>}\' to '
        "configure a backend. Values supported in <value> are those included by "
        "`flwr.common.typing.ConfigsRecordValues`. ",
    )
    parser.add_argument(
        "--enable-tf-gpu-growth",
        action="store_true",
        help="Enables GPU growth on the main thread. This is desirable if you make "
        "use of a TensorFlow model on your `ServerApp` while having your `ClientApp` "
        "running on the same GPU. Without enabling this, you might encounter an "
        "out-of-memory error because TensorFlow by default allocates all GPU memory."
        "Read more about how `tf.config.experimental.set_memory_growth()` works in "
        "the TensorFlow documentation: https://www.tensorflow.org/api/stable.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="When unset, only INFO, WARNING and ERROR log messages will be shown. "
        "If set, DEBUG-level logs will be displayed. ",
    )
    parser.add_argument(
        "--app-dir",
        default="",
        help="Add specified directory to the PYTHONPATH and load"
        "ClientApp and ServerApp from there."
        " Default: current working directory.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="Sets the ID of the run started by the Simulation Engine.",
    )

    return parser
