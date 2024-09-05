from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from .local_average import get_local_average_and_count
import json
import os


class AverageExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        if task_name == "get_local_average_and_count":
            data_dir_path = get_data_dir_path(fl_ctx)
            local_average_and_count = get_local_average_and_count(
                data_dir_path)

            # save local average to local results file
            save_results_to_file(
                local_average_and_count,
                "local_average.json",
                fl_ctx
            )

            outgoing_shareable = Shareable()
            outgoing_shareable["result"] = local_average_and_count
            return outgoing_shareable

        if task_name == "accept_global_average":
            # save global average to local results file
            result = {"global_average": shareable.get("global_average", {})}
            save_results_to_file(
                result,
                "global_average.json",
                fl_ctx
            )
            return Shareable()


def save_results_to_file(results: dict, file_name: str, fl_ctx: FLContext):
    results_dir = get_results_dir_path(fl_ctx)
    print(f"\nSaving results to: {results_dir}\n")
    with open(os.path.join(results_dir, file_name), "w") as f:
        json.dump(results, f)


def get_results_dir_path(fl_ctx: FLContext) -> str:
    """
    Determines the appropriate results directory path for the federated learning application.

    The function first checks for a 'RESULTS_DIR' environment variable. If not found,
    it constructs the path based on conventions for Simulator and POC modes and ensures the directory exists.

    Parameters:
    - fl_ctx: FLContext, providing context for the federated learning client.

    Returns:
    - The path to the results directory as a string.
    """

    # Check for a globally defined results directory first.
    results_dir = os.getenv("RESULTS_DIR")
    if results_dir:
        return results_dir

    # Construct potential paths for Simulator and POC mode.
    job_id = fl_ctx.get_job_id()
    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)

    # first find the base directory, it could be ../../../test_results or ../../../../test_results
    simulator_base_path = os.path.abspath(
        os.path.join(os.getcwd(), "../../../test_results"))
    poc_base_path = os.path.abspath(os.path.join(
        os.getcwd(), "../../../../test_results"))
    simulator_path = os.path.join(simulator_base_path, job_id, site_name)
    poc_path = os.path.join(poc_base_path, job_id, site_name)

    if os.path.exists(simulator_base_path):
        os.makedirs(simulator_path, exist_ok=True)
        return simulator_path
    elif os.path.exists(poc_base_path):
        os.makedirs(poc_path, exist_ok=True)
        return poc_path
    else:
        raise FileNotFoundError(
            "Results directory path could not be determined.")


def get_data_dir_path(fl_ctx: FLContext) -> str:
    """
    Determines the appropriate data directory path for the federated learning application by checking
    if the expected directories for Simulator or POC modes exist.

    It first checks for a 'DATA_DIR' environment variable. If not found,
    it tries to find the data directory based on the conventional paths for Simulator and POC modes.

    Parameters:
    - fl_ctx: FLContext, providing context for the federated learning client.

    Returns:
    - The path to the data directory as a string.
    """

    # Check for a globally defined data directory first.
    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        return data_dir

    # Construct potential paths for Simulator and POC mode.
    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
    simulator_path = os.path.abspath(os.path.join(
        os.getcwd(), "../../../test_data", site_name))
    poc_path = os.path.abspath(os.path.join(
        os.getcwd(), "../../../../test_data", site_name))

    # Check if the Simulator mode path exists, else fall back to the POC mode path if it exists.
    if os.path.exists(simulator_path):
        return simulator_path
    elif os.path.exists(poc_path):
        return poc_path
    else:
        raise FileNotFoundError("Data directory path could not be determined.")
