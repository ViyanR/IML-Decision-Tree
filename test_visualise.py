
import matplotlib.pyplot as plt

def is_leaf(node):
    return "class" in node.keys()

def resolve_overlap(x, y, node_pos, min_separation):
    while any(abs(x - node_x) < min_separation and y==node_y for
            node_x, node_y in node_pos):
            x += min_separation
    return x

def visualise_decision_tree(tree, depth, x=0, y=0, level=0, parent_x=None, parent_y=None, is_left=None, node_positions=set()):
    if tree is None:
        return

    is_leaf_node = is_leaf(tree)

    if parent_x is not None:
        x = parent_x + (-1 if is_left else 1) * 2 ** (depth-level) / 8000
        x = resolve_overlap(x, y, node_positions, 200)
    
    node_positions.add((x,y))

    plt.plot([parent_x, x], [parent_y, y], lw=1, color="red" if is_left else "blue")

    ha = va = "center"
    fontsize = 6
    background_color = "lightgray"
    leaf_background_color = "green"

    if not is_leaf_node:
        plt.text(x, y, f"X{tree['attribute']} < {tree['value']}", ha=ha, va=va, fontsize=fontsize, fontweight="bold", backgroundcolor=leaf_background_color, picker=True)
    else:
        plt.text(x, y, f"{tree['class']}", ha=ha, va=va, fontsize=fontsize, fontweight="bold", backgroundcolor=leaf_background_color, picker=True)
    
        left_child_x = x
        right_child_x = x

        left_child_y = y-2
        right_child_y = y-2

        visualise_decision_tree(tree["left"],depth,left_child_x,left_child_y,level+1,x,y,is_left=True,node_positions=node_positions)
        visualise_decision_tree(tree["right"],depth,right_child_x,right_child_y,level+1,x,y,is_left=False,node_positions=node_positions)

def visualise(tree, depth):
    plt.figure(figsize=(17, 8))
    plt.axis("off")
    visualise_decision_tree(tree, depth)
    plt.show()

if __name__ == "__main__":
    visualise({"attribute": "2", "value": "-23", "left": {"class": "2"}, "right": {"class": "3"}}, 0)