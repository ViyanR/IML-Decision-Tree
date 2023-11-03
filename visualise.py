import matplotlib.pyplot as plt

def visualise_tree(trained_tree, ax=None,indent_x=0.5,indent_y=1, parent_pos=None, width=0.5, node_pos = set()):
    if ax is None:
        ax = plt.gca()
    if "class" in trained_tree.keys():
        annotation =  plt.text(indent_x, indent_y, f"{trained_tree['class']}", ha="center", va="center", fontsize=6, fontweight="bold", backgroundcolor="green", picker=True)
        
        pos = annotation.get_position()
        node_pos.add(pos)
        # print(pos)
        if parent_pos is not None:
             x_coords = [parent_pos[0], pos[0]]
             y_coords = [parent_pos[1], pos[1]]
             ax.plot(x_coords, y_coords, color='blue')
    else:
        annotation = plt.text(indent_x, indent_y, f"{trained_tree['attribute']} < {trained_tree['value']}", ha="center", va="center", fontsize=6, fontweight="bold", backgroundcolor="beige", picker=True)
        pos = annotation.get_position()
        node_pos.add(pos)
        # print(pos)
        if parent_pos is not None:
             x_coords = [parent_pos[0], pos[0]]
             y_coords = [parent_pos[1], pos[1]]
             ax.plot(x_coords, y_coords, color='blue')
        visualise_tree(trained_tree["left"],ax,indent_x-width,indent_y-0.05, pos,width/2, node_pos)
        visualise_tree(trained_tree["right"],ax,indent_x+width,indent_y-0.05, pos,width/2, node_pos)
        
if __name__ == "__main__":
    # visualise_tree({"class":"3"})
    visualise_tree({"attribute": "2", "value": "-23", "left": {"class": "2"}, "right": {"class": "3"}})
    plt.axis('off')
    plt.show()