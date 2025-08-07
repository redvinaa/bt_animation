# BT Animation

A python and opencv based behavior tree visualizer and animation tool. Great for debugging or creating instructional videos.

## Features
- Draws behavior trees from a simple TreeNode structure
- Animates tick propagation, status updates, and overlays
- Exports animations to .mp4 using OpenCV

Nodes are referred to by ID, thus names don't have to be unique and can also change during execution.

### Actions:

- SetStatus: Set color of node
- Tick and TickParallel: Animate tick propagation through the tree, also sets status after tick
- Wait: Pause animation for a specified duration
- RenameNode: Change the name of a node during animation
- ShowOverlay: Blur display and show a message overlay

## Usage:

Everything is implemented in the `visualize_bt` module, which also includes a basic usage example.
For a more complex example, see the `examples/` folder.

### Output
The animation is shown during execution, and a video can also be saved (with filename argument).
The video will be saved in the current working directory with the specified name.

### Requirements
- python 3.8+
- opencv-python
- numpy
- networkx
- (optional) pygraphviz for tree layout — fallback is spring_layout
