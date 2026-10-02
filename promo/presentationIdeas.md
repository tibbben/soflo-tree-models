
###  Introduction (Tim):  approximately 5 minutes

#### Tree Inventory Introduction (slide 1)

For several years IDSC and the UM Libraries (Tim) have partnered with the UM Sustainability Office (Teddy Lhoutellier) to create a tree inventory on the UM Gables Campus.

- always involves student interns
   - initial creation of the platform on ArcGIS Online
   - data collection and maintenance (including use in the Resilience Institute's class - Urban Resilience)
- built upon drone surveys of the campus (Chris Mader)
   - 2019, 2022, 2025, 2026
   - included student interns on Tiling the MagicVerse project 
      - to create tield meshes from the drone survey (partial failure)

#### The problems we approached (still slide 1)

__principal problem:__ _Canned tree detection models do not perform well on south Florida urban forests; nor on landscapes in the Big Cypress National Preserve._  

__NOTE:__ this is not a new problem - ML approaches to tree detection from arial images has a history of at least two decades. Results have varied .... but are slowly improving.

__Side Note:__ the original tree detection for the UM campus used an inverted height map and a water flow model to identify trees (a known approach). It was  ... OK. Then the students corrected the detection errors.

- _first question:_ how can we use AI coding agents to train models on labelled drone survey data
- _second question:_ can we then use these models to recognixze trees in Big Cypress
- _third question:_ how can we responsibly encourage students to use AI in their learning
- _fourth question:_ can the students teach me to use coding agents?

#### The approach

- encourage use of AI coding agents with weekly meetings to share and guide the flow (including medium level subscriptions to claude)
- everything shared through box (training data) and github (code base and explanaitons)
- encourage individual approaches with shared goals
- a ver ... que pasará

### Both:  

Please share links to your current drafts if possible.

basic outline:

- repeat problems/questions briefly (?)
- your approach / method
- your results
- your thinking on what worked and what did not work
- what you learned

some things to think about:

- emphasize what you learned not just ML approaches.   
   - workflows, documentation, GIS, AI integration, etc  
   - both did a good job on reproducibility  
- licensing? what sources were used for code creation? authorship? contact?   
- tiles and chips used interchangeably?  
- what machines were used ... architectures?   
- what observations about the data do you have? How to improve labelling and data collection (very relevant)  

emphasize in the conclusions:  

- what succeded and why
- what failed and why

_Will it be possible to merge the story for this section??_

### Ahsan (tree_detection)

PC and Triton Implementation?  

- perhaps highlight some of the work on Triton?
   - can you give some examples of run times (PC vs Triton)?
- Emphasize the variables you used in the experiments ...
   - resolution
   - canopy size
   - different models
   - ????
- Gables
   - why 'champion'?
   - perhaps better visualization of results (map), with insets and stories?
- Big Cypress
   - did you use the out of the box models? or those trained on the UM data? or both

### Rayan (treedetect)

Mac only implementation?  

- Gables
   - good use of visualiztions in python - perhaps separate a little more?
   - put reported metrics into easy to digest format?
   - perhaps some QGIS visualizations?
   - resolutions used? Other variables tweaked?
- Big Cypress
   - mostly plots of existing data ...?

### Conclusion (all)

- brief literature review
   -  our results compared favorably with recent approaches
   - the problems identified by Ahsan and Rayan are similar to those identified in recent literature  
- while not ground-breaking work, this highlights a need to create local models for local landscapes
- did we learn something about learning? 
   - students are often the best teachers? 
   - interpreting the world together - pedagogy of the oppressed
- Critical Data Science?
   - reproducibility of AI workflows ... the need to review ...
   - AI and learning ... human in the loop ...
   - anything else ???