import os
from re import L
import sys
import time
import json
import numpy as np
import argparse
from enum import IntEnum
from copy import deepcopy

import habitat_sim

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

from structures import Intention, BehaviourGraph
from map_utils import map2MeshFrame, convertMapFrame

class EditType(IntEnum):
    ADD_PLACE_NODE = 0
    DELETE_PLACE_NODE = 1
    ADD_CHANGEPOINT_NODE = 2
    DELETE_CHANGEPOINT_NODE = 3
    ADD_LINE = 4
    DELETE_LINE = 5
    PRINT_NODE_EDGES = 6

def calculateResolution(pathfinder, width):
    lower, upper = pathfinder.get_bounds()
    return abs(upper[2] - lower[2]) / float(width)

class MapVisualiser:
    def __init__(self, scene_dir, dataset_type, graph_dir=""):
        self.scene_dir = scene_dir
        self.dataset_type = dataset_type
        self.setupSim()

        self.bounds = self.sim.pathfinder.get_bounds()
        self.res = 0.1
        self.map_height = self.sim.get_agent(0).state.position[1]
        print("Agent height: ", self.map_height)

        top_down_map = self.sim.pathfinder.get_topdown_view(
            meters_per_pixel=self.res,
            height=self.map_height
            # height=0
        )
        top_down_map_array = np.array(top_down_map).astype(np.uint8) * 255
        self.map = convertMapFrame(top_down_map_array)

        self.intention = None
        self.direction = 0
        self.picked_point = None
        self.delete_overlapping = False
        self.edit_mode = EditType.ADD_PLACE_NODE
        self.quit = False

        # Define keypresses
        self.FINISH = "f"
        self.SAVE = "ctrl+s"

        # Define GUI constants and values
        self.arrow_length = 4
        self.arrow_width = 2.5
        self.node_size = 25
        self.intention_colourmap = {Intention.LEFT: 'b', Intention.RIGHT: 'r', Intention.FORWARD: '0.6'}
        self.intention_textmap = {Intention.LEFT: 'LEFT', Intention.RIGHT: 'RIGHT', Intention.FORWARD: 'FORWARD'}
        self.n_directions = 2

        self.vis_direction = self.n_directions

        # Set up graph data structure
        self.graph = BehaviourGraph(self.scene_dir)
        if graph_dir == "":
            self.draw = False
        else:
            self.loadData(graph_dir)
            self.draw = True

        # Set up figure
        plt.ion()
        plt.rcParams['keymap.save'].remove('s')
        plt.rcParams['keymap.save'].remove('ctrl+s')
        plt.rcParams['keymap.fullscreen'].remove('f')

        self.fig, self.axs = plt.subplots(1, 1)
        # self.fig.set_figwidth(16)
        # plt.tight_layout()

        self.binding_id = self.fig.canvas.callbacks.connect('button_press_event', self.onClick)
        self.fig.canvas.callbacks.connect('key_press_event', self.onPress)

    ### Main rendering and interaction loop ###
    def hasEvent(self):
        return (self.draw or self.picked_point is not None)

    def render(self):
        self.axs.imshow(self.map, origin="lower")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        line_endpoints = []
        while True:
            if not self.hasEvent():
                time.sleep(0.01)
            else:
                self.draw = False

                # Handle event
                if self.quit:
                    break

                if self.picked_point is not None:
                    if self.edit_mode == EditType.ADD_PLACE_NODE:
                        self.addNode(place=True)
                    elif self.edit_mode == EditType.ADD_CHANGEPOINT_NODE:
                        self.addNode(place=False)
                    elif self.edit_mode == EditType.DELETE_PLACE_NODE or self.edit_mode == EditType.DELETE_CHANGEPOINT_NODE:
                        self.deleteNode()
                    elif self.edit_mode == EditType.PRINT_NODE_EDGES:
                        self.printNodeEdges()
                    elif self.edit_mode == EditType.ADD_LINE:
                        line_endpoints.append(self.picked_point)
                        line_endpoints = [] if self.addLine(line_endpoints) else line_endpoints
                    else:
                        line_endpoints.append(self.picked_point)
                        line_endpoints = [] if self.deleteLine(line_endpoints) else line_endpoints
                    self.picked_point = None

                # Update visualisation
                self.axs.clear()
                self.axs.imshow(self.map, origin="lower")

                graph_nodes_pix = list(map(lambda node: node[0], self.graph.graph_nodes))

                if len(graph_nodes_pix) > 0:
                    colours = [("#fa8072" if node[2] else "g") for node in self.graph.graph_nodes]
                    graph_nodes_pix = np.array(graph_nodes_pix)
                    self.axs.scatter(graph_nodes_pix[:,0], graph_nodes_pix[:,1], c=colours, s=self.node_size, picker=True)

                    for idx in range(len(self.graph.graph_nodes)):
                        self.axs.annotate(idx, (graph_nodes_pix[idx, 0], graph_nodes_pix[idx, 1]), color="green")

                if len(self.graph.graph_edges) > 0:
                    for src_id, dst_id, edge_int, edge_dir in self.graph.graph_edges:
                        if not (self.vis_direction == self.n_directions or edge_dir == self.vis_direction):
                            continue
                        endpoints = np.array([graph_nodes_pix[src_id], graph_nodes_pix[dst_id]])
                        if edge_dir == 0:
                            overhang = 0.5
                            linestyle = ':'
                        else:
                            overhang = 0.
                            linestyle = '-'
                        self.axs.arrow(*endpoints[0], *(endpoints[1] - endpoints[0]),
                            color=self.intention_colourmap[edge_int], overhang=overhang,
                            length_includes_head=True, head_length=self.arrow_length, 
                            head_width=self.arrow_width, linestyle=linestyle)
                self.fig.canvas.draw()

            self.fig.canvas.flush_events()

        self.sim.close()

    ### Functions for editing behaviour graph ###
    def addNode(self, place=False):
        pos_x = self.picked_point[0] * self.res + self.bounds[0][0]
        pos_y = self.picked_point[1] * self.res - self.bounds[1][2]

        pos_map_frame = [pos_x, self.map_height, pos_y]
        pos_mesh_frame = map2MeshFrame(pos_map_frame)
        print(">>> ", pos_map_frame)
        print("### ", pos_mesh_frame)
        if self.sim.pathfinder.is_navigable(np.array(pos_mesh_frame)):
            self.graph.graph_nodes.append((self.picked_point, pos_map_frame, place))
        else:
            print("Select a point in navigable space!")

    def deleteNode(self):
        updated_graph_edges = []
        for src_id, dst_id, edge_int, edge_dir in self.graph.graph_edges:
            if src_id != self.picked_point and dst_id != self.picked_point:
                updated_graph_edges.append([src_id - 1 if src_id > self.picked_point else src_id,
                                            dst_id - 1 if dst_id > self.picked_point else dst_id,
                                            edge_int, edge_dir])
        del self.graph.graph_nodes[self.picked_point]
        self.graph.graph_edges = updated_graph_edges

    def addLine(self, line_endpoints):
        if len(line_endpoints) == 2:
            if line_endpoints[0] != line_endpoints[1]:
                if self.intention is None:
                    print("Select intention before adding line")
                    return True
                line_endpoints.append(self.intention)
                line_endpoints.append(self.direction)
                self.graph.graph_edges.append(line_endpoints)
                print("Adding line with intention ", self.intention_textmap[self.intention])
                print(line_endpoints)
            return True
        return False

    def deleteLine(self, line_endpoints):
        if len(line_endpoints) == 2:
            print(line_endpoints)
            for n, (src_id, dst_id, _, _) in enumerate(self.graph.graph_edges):
                if src_id == line_endpoints[0] and dst_id == line_endpoints[1]:
                    del self.graph.graph_edges[n]
                    break
            return True
        return False

    def printNodeEdges(self):
        print("### Outgoing edges of node: ", self.picked_point)
        for edge in self.graph.graph_edges:
            if edge[0] == self.picked_point:
                print("Edge: ", edge)

    ### GUI callbacks ###   
    def onMove(self, event):
        # get the x and y pixel coords
        x, y = event.x, event.y
        if event.inaxes:
            ax = event.inaxes  # the axes instance
            print('data coords %f %f' % (event.xdata, event.ydata))

    def onClick(self, event):
        print('Received click event')
        if event.button is MouseButton.LEFT:
            x, y = event.x, event.y
            if event.inaxes == self.axs:
                print('Selected point: ', event.xdata, event.ydata)
                self.picked_point = np.array([event.xdata, event.ydata])
            else:
                print('Please only select point on the top-down map')

    def onPick(self, event):
        if len(event.ind) > 1:
            print("Overlapping points picked!")
            for ind in event.ind:
                print(ind, ": ", self.graph.graph_nodes[ind])
            if self.delete_overlapping:
                print("Currently set to delete overlapping! Will delete ", event.ind[0])
                self.picked_point = event.ind[0]
                self.delete_overlapping = False
            else:
                print("Not deleting any point. To delete the overlapping points one-by-one, toggle 'd'.")
                self.picked_point = None
        else:
            ind, = event.ind
            self.picked_point = ind
            print("Picked point: ", ind, self.graph.graph_nodes[ind])

    def onPress(self, event):
        sys.stdout.flush()

        if event.key == self.SAVE:
            self.writeData()
        elif event.key == self.FINISH:
            self.quit = True
        elif event.key == "d":
            self.delete_overlapping = True
        elif event.key == "v":
            self.vis_direction = (self.vis_direction + 1) % (self.n_directions + 1)
            self.draw = True
            print("Toggled to visualise direction: ", self.vis_direction)
        elif event.key == "0":
            self.toggleDirection()
        elif event.key == "1":
            print("Switch edit mode: Adding place nodes.")
            self.edit_mode = EditType.ADD_PLACE_NODE
            plt.disconnect(self.binding_id)
            self.binding_id = self.fig.canvas.callbacks.connect("button_press_event", self.onClick)
        elif event.key == "2":
            print("Switch edit mode: Deleting place nodes.")
            self.edit_mode = EditType.DELETE_PLACE_NODE
            plt.disconnect(self.binding_id)
            self.binding_id = self.fig.canvas.callbacks.connect("pick_event", self.onPick)
        elif event.key == "3":
            print("Switch edit mode: Adding changepoint nodes.")
            self.edit_mode = EditType.ADD_CHANGEPOINT_NODE
            plt.disconnect(self.binding_id)
            self.binding_id = self.fig.canvas.callbacks.connect("button_press_event", self.onClick)
        elif event.key == "4":
            print("Switch edit mode: Deleting changepoint nodes.")
            self.edit_mode = EditType.DELETE_CHANGEPOINT_NODE
            plt.disconnect(self.binding_id)
            self.binding_id = self.fig.canvas.callbacks.connect("pick_event", self.onPick)
        elif event.key == "5":
            print("Switch edit mode: Adding edges.")
            self.edit_mode = EditType.ADD_LINE
            plt.disconnect(self.binding_id)
            self.binding_id = self.fig.canvas.callbacks.connect("pick_event", self.onPick)
        elif event.key == "6":
            print("Switch edit mode: Deleting edges.")
            self.edit_mode = EditType.DELETE_LINE
            plt.disconnect(self.binding_id)
            self.binding_id = self.fig.canvas.callbacks.connect("pick_event", self.onPick)
        elif event.key == "9":
            self.edit_mode = EditType.PRINT_NODE_EDGES
            plt.disconnect(self.binding_id)
            self.binding_id = self.fig.canvas.callbacks.connect("pick_event", self.onPick)
        elif event.key == "left":
            self.intention = Intention.LEFT
            print("Switched to intention: LEFT")
        elif event.key == "right":
            self.intention = Intention.RIGHT
            print("Switched to intention: RIGHT")
        elif event.key == "up":
            self.intention = Intention.FORWARD
            print("Switched to intention: FORWARD")
        elif event.key == "m":
            self.saveMap()
            print("Saved map to file")
        else:
            print("Invalid keypress: ", event.key)

    def toggleDirection(self):
        self.direction = (self.direction + 1) % self.n_directions
        print("Toggled direction to: ", self.direction)

    # I/O
    def writeData(self):
        self.graph.writeGraph()

    def loadData(self, filepath):
        self.graph.loadGraph(filepath)

    def saveMap(self):
        basename = os.path.basename(self.scene_dir)
        dirname = os.path.dirname(self.scene_dir)
        scenename = basename.split(".")[0]
        filename = scenename + "_map.npz"
        outfile = os.path.join(dirname, filename)

        # Note that the Habitat simulator flips the y-axis
        # when loading scenes. The bounds saved are the Habitat
        # bounds, which are the reverse and negation of the
        # original bounds in the original scene/mesh coordinate
        # frame.
        np.savez(
            outfile,
            map=self.map,
            bounds=self.bounds,
            res=self.res,
            map_height=self.map_height
        )


    ### Set up simulator ###
    def setupSim(self):
        config = habitat_sim.SimulatorConfiguration()
        config.scene_id = self.scene_dir

        agent_config = habitat_sim.AgentConfiguration()
        agent_config.sensor_specifications = self.setupSensors()
        sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent_config]))
        
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        if not sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=False):
            print("Failed to recompute navmesh")
            sys.exit(0)

        if not sim.pathfinder.is_loaded:
            print("Sim pathfinder not loaded!")
            sys.exit(0)

        self.sim = sim

    def setupSensors(self):
        left_rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
        left_rgb_sensor.uuid = "left_rgb"
        left_rgb_sensor.resolution = [256, 256]
        left_rgb_sensor.position = 1.5 * habitat_sim.geo.UP + 0.15 * habitat_sim.geo.LEFT + 0.05 * habitat_sim.geo.BACK
        left_rgb_sensor.orientation = [0., np.pi/2, 0.]

        return [left_rgb_sensor]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, help="Path to scene file",
                        # default="/mnt/data/datasets/gibson/2d3ds_for_gibson/area3/area3_rotated.glb")
                        # default="/home/joel/research/behaviour_mapping/data/area3/area3_rotated.glb")
                        default="/Users/joel/Research/data/area3/area3_rotated.glb")
    parser.add_argument("--dataset_type", type=str, help="Dataset used: gibson, 2d3ds, habitat",
                        default="2d3ds")
    args = parser.parse_args()

    # vis = MapVisualiser(
    #     args.scene_dir,
    #     args.dataset_type
    #     )
    vis = MapVisualiser(
        args.scene_dir,
        args.dataset_type,
        # "/mnt/data/datasets/gibson/2d3ds_for_gibson/area3/area3_rotated_graph.json"
        # "/home/joel/research/behaviour_mapping/data/area3/area3_rotated_graph.json"
        "/Users/joel/Research/data/area3/area3_rotated_graph.json"
    )
    vis.render()
