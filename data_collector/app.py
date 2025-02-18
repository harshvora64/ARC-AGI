from flask import Flask, render_template, request, jsonify, url_for
import os
import json
from matplotlib.colors import ListedColormap, NoNorm
import matplotlib
matplotlib.use('Agg')  # Prevents the need for a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import scipy.optimize
import pickle

app = Flask(__name__)

# Directory for generated images
IMAGE_DIR = os.path.join("static", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Load dataset JSON files
dataset_dir = {'train': '../data/training', 'eval': '../data/evaluation'}
dataset = {'train': {}, 'eval': {}}
for split in dataset_dir:
    for root, dirs, files in os.walk(dataset_dir[split]):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    dataset[split][file] = data

# Default file selection
default_filename = 'f8a8fe49.json'

def update_datasets(chosen_filename):
    training_train_inputs = [np.array(x['input']) for x in dataset['train'][chosen_filename]['train']]
    training_train_outputs = [np.array(x['output']) for x in dataset['train'][chosen_filename]['train']]
    training_test_inputs = [np.array(x['input']) for x in dataset['train'][chosen_filename]['test']]
    training_test_outputs = [np.array(x['output']) for x in dataset['train'][chosen_filename]['test']]
    return training_train_inputs, training_train_outputs, training_test_inputs, training_test_outputs

def print_grid(grid, image_filename, pos=(0,0)):
    colors = [
        "#000000",  # black
        "#E0E0E0",  # pastel grey
        "#EEBFA9",  # pastel orange
        "#CCFF99",  # pastel lime
        "#AFEEEE",  # pastel cyan
        "#ADFFAD",  # pastel green
        "#EEE8AA",  # pastel yellow
        "#EEA9A9",  # pastel red
        "#AEAEEE",  # pastel blue
        "#B0C4DE",  # pastel purple
        "#EEAEE0",  # pastel something
        "#EEA9B8"   # pastel pink
    ]
    unique_values = map(round, sorted(np.unique(grid)))
    cmap = ListedColormap([colors[val] for val in unique_values], name='custom_cmap')
    x_size = max(1, grid.shape[1])
    y_size = max(1, grid.shape[0])
    figsize = (x_size / 2, y_size / 2)
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.pcolormesh(grid, cmap=cmap, edgecolors='white', linewidth=0.1)
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            plt.text(j + 0.5, i + 0.5, f"{j+pos[1]},{i+pos[0]}", ha='center', va='center', 
                     fontsize=12, color='white', fontweight='bold')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.gcf().set_size_inches(figsize)
    plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def extract_selected_objects(grid, diag=False, same_color=True, black=False, full_grid = False):
    if full_grid:
        u = np.unique(grid)
        u = u[u != 0]
        if len(u) == 1:
            col = u[0]
        else:
            col = -1
        return[((0,0), col, grid)]
    
    if black:
        same_color = True
        diag = False
    objects = []
    visited = np.zeros(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if visited[i][j] == 0:
                obj = np.zeros(grid.shape)
                stack = [(i, j)]
                visited[i][j] = 1
                color = grid[i][j]
                if black and color != 0:
                    continue
                if color == 0 and not black:
                    continue
                while stack:
                    x, y = stack.pop()
                    if not black:
                        obj[x][y] = grid[x][y]
                    else:
                        obj[x][y] = 11
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            condn = (x+dx >= 0 and x+dx < grid.shape[0] and 
                                     y+dy >= 0 and y+dy < grid.shape[1] and 
                                     visited[x+dx][y+dy] == 0)
                            if not diag:
                                condn = condn and (dx == 0 or dy == 0)
                            if same_color:
                                condn = condn and grid[x+dx][y+dy] == grid[x][y]
                            if black:
                                condn = condn and grid[x+dx][y+dy] == 0
                            if condn:
                                stack.append((x+dx, y+dy))
                                visited[x+dx][y+dy] = 1
                minx, maxx = grid.shape[0], -1
                miny, maxy = grid.shape[1], -1
                for i1 in range(grid.shape[0]):
                    for j1 in range(grid.shape[1]):
                        if obj[i1][j1] != 0:
                            minx = min(minx, i1)
                            maxx = max(maxx, i1)
                            miny = min(miny, j1)
                            maxy = max(maxy, j1)
                obj = obj[minx:maxx+1, miny:maxy+1]
                objects.append(((minx, miny), color, obj))
    return sorted(objects, key=lambda x: (x[0], x[1]))

def extract_all_possible_objects(grid, key=lambda x: (x[0], x[1])):
    objects = (extract_selected_objects(grid, diag=True, same_color=True) +
               extract_selected_objects(grid, diag=False, same_color=True) +
               extract_selected_objects(grid, diag=False, same_color=False, black=True) +
               extract_selected_objects(grid, full_grid=True))
    objects = sorted(objects, key=key)
    unique_objects = []
    for i in range(len(objects)):
        if i == 0 or not (np.array_equal(objects[i][2], objects[i-1][2]) and 
                           objects[i][0] == objects[i-1][0] and 
                           objects[i][1] == objects[i-1][1]):
            unique_objects.append((objects[i][0], objects[i][1], np.array(objects[i][2])))
    return sorted(unique_objects, key=key)

def process_grid(grid, image_filename):
    print_grid(grid, image_filename)

def process_object(obj, image_filename):
    print_grid(obj[2], image_filename, obj[0])

def generate_data(file_key):
    # Update datasets based on the chosen file key
    training_train_inputs, training_train_outputs, training_test_inputs, training_test_outputs = update_datasets(file_key)
    input_grids = training_train_inputs
    output_grids = training_train_outputs
    extra_input_grid = training_test_inputs[0]
    
    pairs = []
    local_object_mapping = {}
    for idx, input_grid in enumerate(input_grids):
        input_image_file = os.path.join(IMAGE_DIR, f"input_{idx}.png")
        process_grid(input_grid, input_image_file)
        
        output_image_file = os.path.join(IMAGE_DIR, f"output_{idx}.png")
        process_grid(output_grids[idx], output_image_file)
        
        objects = extract_all_possible_objects(input_grid)
        object_data = []
        for j, obj in enumerate(objects):
            obj_id = f"pair_{idx}_object_{j}"
            object_image_file = os.path.join(IMAGE_DIR, f"pair_{idx}_object_{j}.png")
            process_object(obj, object_image_file)
            object_data.append({
                "id": obj_id,
                "position": obj[0],
                "color": obj[1],
                "image": object_image_file
            })
            local_object_mapping[obj_id] = obj
        pairs.append({
            "id": f"pair_{idx}",
            "input_image": input_image_file,
            "output_image": output_image_file,
            "objects": object_data
        })
    
    extra_input_image = os.path.join(IMAGE_DIR, "input_extra.png")
    process_grid(extra_input_grid, extra_input_image)
    extra_objects = extract_all_possible_objects(extra_input_grid)
    extra_object_data = []
    for j, obj in enumerate(extra_objects):
        obj_id = f"pair_extra_object_{j}"
        object_image_file = os.path.join(IMAGE_DIR, f"pair_extra_object_{j}.png")
        process_object(obj, object_image_file)
        extra_object_data.append({
            "id": obj_id,
            "position": obj[0],
            "color": obj[1],
            "image": object_image_file
        })
        local_object_mapping[obj_id] = obj
    extra_pair = {
        "id": "pair_extra",
        "input_image": extra_input_image,
        "output_image": None,
        "objects": extra_object_data
    }
    
    return pairs, extra_pair, local_object_mapping

# Define the available categories
CATEGORIES = ["Category A", "Category B", "Category C", "Category D", "Category E"]

@app.route("/")
def index():
    file_key = request.args.get("filename", default=default_filename)
    pairs, extra_pair, local_mapping = generate_data(file_key)
    available_files = list(dataset['train'].keys())
    return render_template("index.html", pairs=pairs, extra_pair=extra_pair, categories=CATEGORIES, chosen_filename=file_key, available_files=available_files)

@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()
    file_key = data.get("filename", default_filename)
    assignments_received = data.get("assignments", {})
    # Re-run generate_data to get the mapping for the chosen file
    _, _, local_mapping = generate_data(file_key)
    assignments_with_objects = {}
    for pair_id, cat_map in assignments_received.items():
        assignments_with_objects[pair_id] = {}
        for cat, obj_ids in cat_map.items():
            assignments_with_objects[pair_id][cat] = [local_mapping[obj_id] for obj_id in obj_ids if obj_id in local_mapping]
    with open(f"assignments/{file_key[:-5]}.pkl", "wb") as f:
        pickle.dump(assignments_with_objects, f)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
