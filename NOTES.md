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
- Menu with no selection? an update all entry would allow to add new variables added under alias
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