To develop:  
- Optional display alternatives for objects:
  - Default: str if exist, else prettify
  - Prettify only
  - Str only
  - Repr only  
  Max length parameter
- Drop files to import
- Fix drag for object key dicts
- Bug: set a variable under alias.child, update, set it to none, can't update
- Bug: "AttributeError: Can't pickle local object 'create_namedtuple_with_type_equality.<locals>.NamedTupleWithCustomEquality'" when dragging 
- Bug: froze on .mat export from a list of dicts containing sva ._asdict() results; sva.variable_viewer.exporter.save_as_mat_single(name='result',value=sva.tmp_video_sva,file_path=r'C:\Users\olivi\Documents\code\evm\data\potence\results\result1.mat') gives some string only
- Menu with no selection? an update all entry would allow to add new variables added under alias
- Tree: slow with long lists. Expand should be limited to n elements
- Variables preview:
  - change ... placement (in array, only if more elements)
  - list of NamedTupleWithCustomEquality in sva: why does it not display str representation?

To experiment:
- AI agent plot:  
Pass item structure metadata plus plot prompt to AI agent.  
AI agent checks which plots are compatible with the data.  
AI agent proposes best plot types to user.  
User validates.  
AI agent generates plot code and runs plot function.