# To develop:  
- Optional display alternatives for objects:
  - Default: str if exist, else prettify
  - Prettify only
  - Str only
  - Repr only  
  Max length parameter
- Drop files to import
- Bug: froze on .mat export from a list of dicts containing sva ._asdict() results; sva.variable_viewer.exporter.save_as_mat_single(name='result',value=sva.tmp_video_sva,file_path=r'C:\Users\olivi\Documents\code\evm\data\potence\results\result1.mat') gives some string only
- Tree: slow with long lists. Expand should be limited to n elements
- Variables preview:
  - change ... placement (in array, only if more elements)
  - list of NamedTupleWithCustomEquality in sva: why does it not display str representation?

# To experiment:
- AI agent plot:  
Pass item structure metadata plus plot prompt to AI agent.  
AI agent checks which plots are compatible with the data.  
AI agent proposes best plot types to user.  
User validates.  
AI agent generates plot code and runs plot function.

# Issues:  

1. Non blocking error message:  

From setup_console.refresh_after_execute  

                except Exception as err:
                    logger.exception("Error during conditional refresh: %s", err)

2025-05-11 13:39:46,617 [DEBUG] var_view.variable_viewer.console: Executed command: c.processed(c.input_data['Sinop_UG1_2024_11_T_167MW_trip.uff']['track_1'],'x4')
2025-05-11 13:39:46,617 [INFO] var_view.variable_viewer.console: Parameter 'processed' does not exist. Refreshing view.
2025-05-11 13:39:46,617 [ERROR] var_view.variable_viewer.console: Error during conditional refresh: 'VariableViewer' object is not callable
2025-05-11 13:39:46,622 [DEBUG] traitlets: execute_input: {'code': "c.processed(c.input_data['Sinop_UG1_2024_11_T_167MW_trip.uff']['track_1'],'x4')", 'execution_count': 6}
2025-05-11 13:39:46,622 [DEBUG] traitlets: execute_reply: {'status': 'ok', 'execution_count': 6, 'user_expressions': {}, 'payload': []}
