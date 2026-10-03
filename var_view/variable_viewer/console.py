# var_view/variable_viewer/console.py

import logging
import re
import threading
import time

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

logger = logging.getLogger(__name__)


def _probe_iopub(kernel, label):
    """Log whether scheduled IOPub work is actually processed.

    The probe is deliberately non-blocking so it does not wait on the IOPub
    thread or materially alter console startup timing. A healthy IOPub thread
    should process the callback almost immediately. If the callback is delayed
    or never logged, the thread is alive but not servicing scheduled work.
    """
    try:
        iopub = kernel.iopub_thread
        thread = iopub.thread
        scheduled_at = time.perf_counter()

        logger.warning(
            "IOPub probe scheduled [%s]: alive=%s ident=%s current_ident=%s",
            label,
            thread.is_alive() if thread is not None else None,
            thread.ident if thread is not None else None,
            threading.current_thread().ident,
        )

        def mark_processed():
            logger.warning(
                "IOPub probe processed [%s]: latency=%.3fs "
                "processor_ident=%s",
                label,
                time.perf_counter() - scheduled_at,
                threading.current_thread().ident,
            )

        iopub.schedule(mark_processed)
    except Exception:
        logger.exception("IOPub probe failed [%s].", label)


def _log_step(label, started_at):
    logger.warning(
        "Console setup step [%s] completed in %.3fs",
        label,
        time.perf_counter() - started_at,
    )


class ConsoleManager:
    def __init__(self, data_source, alias, refresh_callback):
        self.data_source = data_source
        self.alias = alias
        self.refresh_callback = refresh_callback
        self.console_window = None
        self.kernel_manager = None
        self.kernel_client = None
        self.setup_console()

    def setup_console(self):
        try:
            setup_started_at = time.perf_counter()
            logger.warning(
                "Console setup started: current_ident=%s",
                threading.current_thread().ident,
            )

            step_started_at = time.perf_counter()
            self.kernel_manager = QtInProcessKernelManager()
            _log_step("kernel manager creation", step_started_at)

            step_started_at = time.perf_counter()
            self.kernel_manager.start_kernel()
            _log_step("start_kernel", step_started_at)

            kernel = self.kernel_manager.kernel
            _probe_iopub(kernel, "after start_kernel")

            step_started_at = time.perf_counter()
            kernel.gui = "qt"
            _log_step("kernel.gui assignment", step_started_at)
            _probe_iopub(kernel, "after kernel.gui assignment")

            step_started_at = time.perf_counter()
            self.kernel_client = self.kernel_manager.client()
            _log_step("kernel client creation", step_started_at)
            _probe_iopub(kernel, "after client creation")

            step_started_at = time.perf_counter()
            self.kernel_client.start_channels()
            _log_step("start_channels", step_started_at)
            _probe_iopub(kernel, "after start_channels")

            step_started_at = time.perf_counter()
            console = RichJupyterWidget()
            _log_step("RichJupyterWidget creation", step_started_at)
            _probe_iopub(kernel, "after widget creation")

            step_started_at = time.perf_counter()
            console.kernel_manager = self.kernel_manager
            _log_step("console kernel_manager assignment", step_started_at)
            _probe_iopub(kernel, "after kernel_manager assignment")

            step_started_at = time.perf_counter()
            console.kernel_client = self.kernel_client
            _log_step("console kernel_client assignment", step_started_at)
            _probe_iopub(kernel, "after kernel_client assignment")

            step_started_at = time.perf_counter()
            self.console_window = QWidget()
            self.console_window.setWindowTitle("Console")
            layout = QVBoxLayout(self.console_window)
            layout.addWidget(console)
            self.console_window.resize(600, 960)
            self.console_window.show()
            _log_step("console window creation/show", step_started_at)
            _probe_iopub(kernel, "after console show")

            if not hasattr(kernel, 'shell'):
                logger.exception("Kernel does not have a 'shell' attribute.")
                return

            shell = kernel.shell

            if not hasattr(shell, 'events'):
                logger.exception("Kernel shell does not have an 'events' attribute.")
                return

            step_started_at = time.perf_counter()
            shell.push({self.alias: self.data_source})
            _log_step("shell.push data source", step_started_at)
            _probe_iopub(kernel, "after shell.push")

            # Define the event handler
            def refresh_after_execute(result):
                """
                Event handler triggered after a cell is executed.

                Parameters:
                - result: An ExecutionResult object containing execution details.
                """
                try:
                    # Extract the executed cell's source code
                    cell = result.info.raw_cell.strip()
                    logger.debug("Executed command: %s", cell)

                    # Check if the command starts with f"{alias}."
                    if cell.startswith(f"{self.alias}."):
                        # Extract the parameter being accessed or assigned
                        param_match = re.match(
                            rf"{re.escape(self.alias)}\.([A-Za-z_][A-Za-z0-9_]*)", cell)
                        if param_match:
                            param_name = param_match.group(1)
                            full_param_name = f"{param_name}"

                            # Check if the parameter already exists in the viewer
                            if self.refresh_callback.has_variable(full_param_name):
                                logger.debug("Parameter '%s' already exists. No "
                                             "refresh needed.", full_param_name)
                            else:
                                logger.info("Parameter '%s' does not exist. "
                                            "Refreshing view.", full_param_name)
                                self.refresh_callback()
                        else:
                            logger.debug("Could not parse parameter from command: %s",
                                         cell)
                    else:
                        logger.debug("Command does not start with '%s': %s",
                                     self.alias, cell)
                except Exception as err:
                    logger.exception("Error during conditional refresh: %s", err)

            # Register the event handler with post_run_cell
            try:
                step_started_at = time.perf_counter()
                shell.events.register('post_run_cell', refresh_after_execute)
                _log_step("post_run_cell registration", step_started_at)
                logger.info("Registered 'post_run_cell' event handler.")
            except AttributeError as e:
                logger.exception("Failed to register event handler: %s", e)

            _probe_iopub(kernel, "setup complete")
            logger.warning(
                "Console setup completed in %.3fs",
                time.perf_counter() - setup_started_at,
            )
            logger.info("Console window opened and '%s' injected.", self.alias)
        except Exception as e:
            logger.exception("Failed to set up console: %s", e)
