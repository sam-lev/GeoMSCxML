# Standard library imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import copy
# Third party imports
import imageio
from PIL import Image

# Local application
#from topoml.topology.utils import is_ridge_arc, is_valley_arc
from topoml.image.utils import make_mc_arc_mask

#print("%%%%%%%%%%%%%%%%%%%%%% ", os.getcwd())

class MSCNode:
    def __init__(self):
        self.arcs = set()
        self.z = 0
        self.degree = 0
        self.xy = (0,0)
        self.z = 0
        self.index = None
        self.cellid = None

    def read_from_line(self, line):
        tmplist = line.split(",")
        self.cellid = int(tmplist[0])
        self.index = None #int(tmplist[1])
        #self.value = float(tmplist[2])
        #self.boundary = int(tmplist[3])
        self.xy = (float(tmplist[1]), float(tmplist[2]))
        self.z = 0

    def add_arc(self, arc):
        self.arcs.add(arc)
        self.degree = len(self.arcs)

class MLArc:
    def __init__(self):
        self.dim = 0
        self.gid = 0
        self.gid_map = {}

    def __group_xy(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def read_ml_graph_geom_line(self, line, labeled=False):
        tmplist = line.split(" ")
        self.index = int(tmplist[0])
        self.gid = self.index
        self.dim = int(tmplist[1])
        self.line = [
                i for i in self.__group_xy([float(i) for i in tmplist[2:]])
            ] #read the rest of the the points in the arc as xy tuples
        self.gid_map[self.gid] = self.line

class MSCArc:
    def __init__(self):
        self.nodes = []
        self.line = []
        self.label_accuracy = None
        self.partition = None
        self.prediction = None
        self.z = 0
        self.key = None
        self.test = None

        self.dim = 0
        self.gid = 0
        self.gid_map = {}
        self.ground_truth = 0

        self.ogKey = None
        self.ogIndex =None
        self.ogNodeIds = None

        self.intersection_arc = 0

        self.exterior = 0

        self.node_ids = []

    def __group_xy(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def read_from_line(self, line, labeled=False):
        tmplist = line.split(",")
        self.index = int(tmplist[0])
        if labeled:
            self.label_accuracy = float(tmplist[3])
            self.node_ids = [int(tmplist[1]), int(tmplist[2])]
            self.line = [
                i for i in self.__group_xy([float(i) for i in tmplist[4:]])
            ]  # read the rest of the the points in the arc as xy tuples
        else:
            self.node_ids = [int(tmplist[1]), int(tmplist[2])]
            self.line = [
                i for i in self.__group_xy([float(i) for i in tmplist[3:]])
            ] #read the rest of the the points in the arc as xy tuples



class GeoMSC:
    def __init__(self, geomsc=None):
        self.nodes = {}
        self.arcs = []
        self.geomsc = geomsc
        self.msc = geomsc
        self.arc_map = []
        self.selected_arc_map = []
        self.arc_dict = {}
        self.test_geomsc = False
        self.union = 0
        self.z = 0
        self.geomsc_dict = {}
        self.geomsc_dict[self.z] = geomsc

        self.gid_map = {}

        self.max_degree = 0


    def make_arc_id(self,a):
        return tuple(a.node_ids) + (len(a.line),)
    # return true if b contained in a
    def nested_arcs(self,a, b):
        node_match = tuple(a.node_ids) == tuple(b.node_ids)
        nested = not set(a.line).isdisjoint(set(b.line)) #set(a.line).intersection(set(b.line))
        # contains partial subset, non disjoint : not set(a.line).isdisjoint(set(b.line))
        # complete subset :  set(a.line).intersection(set(b.line))
        #is_nested = len(nested) >= int( len(set(a.line))*(0.75) )
        return bool(node_match or nested)

    def read_from_file(self, fname_base, labeled=False):
        nodesname = fname_base + ".nodes.txt"
        arcsname = fname_base + ".arcs.txt"
        node_file = open(nodesname, "r")
        nodes_lines = node_file.readlines()
        node_file.close()
        for l in nodes_lines:
            n = MSCNode()
            n.read_from_line(l)
            self.nodes[n.cellid] = n
        arcs_file = open(arcsname, "r")
        arcs_lines = arcs_file.readlines()
        arcs_file.close()
        for l in arcs_lines:
            a = MSCArc()
            a.read_from_line(l, labeled)
            n1 = self.nodes[a.node_ids[0]]
            n2 = self.nodes[a.node_ids[1]] 
            n1.index = a.index
            n2.index = a.index
            n1.add_arc(a)
            n2.add_arc(a)
            if n1.degree > self.max_degree:
                self.max_degree = n1.degree
            if n2.degree > self.max_degree:
                self.max_degree = n2.degree
            a.nodes = [n1, n2]
            self.arcs.append(a)
            key = self.make_arc_id(a)
            a.key = key

            a.ogKey = key
            a.ogIndex = a.index
            a.ogNodeIds = a.node_ids

            self.arc_dict[key] = a

    def read_from_geo_file(self, fname_base, labeled=False):
        nodesname = fname_base + ".mlg_nodes.txt"
        arcsname = fname_base + ".mlg_edges.txt"
        geoname = fname_base + ".mlg_geom.txt"
        #node_file = open(nodesname, "r")
        #nodes_lines = node_file.readlines()
        #node_file.close()
        #for l in nodes_lines:
        #    n = MSCNode()
        #    n.read_from_line(l)
        #    self.nodes[n.cellid] = n
        geo_file = open(geoname, "r")
        geo_lines = geo_file.readlines()
        geo_file.close()
        cellid = 0
        arcid = 0
        for l in geo_lines:
            mla = MLArc()
            mla.read_ml_graph_geom_line(l, labeled)
            if mla.dim == 1:
                a = MSCArc()
                a.line = mla.line
                a.index = arcid
                arcid += 1
                self.gid_map[mla.gid] = a
            if mla.dim == 0:
                n = MSCNode()
                n.xy = mla.line[0]
                n.cellid = cellid
                cellid += 1
                self.gid_map[mla.gid] = n
        edge_file = open(arcsname, "r")
        edge_lines = edge_file.readlines()
        edge_file.close()   # pull node since reading line /dual graph
        for l in edge_lines:
            tmplist = l.split(' ')
            node = self.gid_map[int(tmplist[0])]
            arc_1 = self.gid_map[int(tmplist[1])]
            arc_2 = None

            node.add_arc(arc_1)
            node.index = arc_1.index
            arc_1.node_ids.append(node.cellid)
            arc_1.node_ids = list(set(arc_1.node_ids))
            if int(tmplist[2]) != -1:
                arc_2 = self.gid_map[int(tmplist[2])]
                node.add_arc(arc_2)
                arc_2.node_ids.append(node.cellid)
                arc_2.node_ids = list(set(arc_2.node_ids))

            self.nodes[node.cellid] = node


            if node.degree > self.max_degree:
                self.max_degree = node.degree

            arc_1.nodes.append(node)
            arc_1.nodes = list(set(arc_1.nodes))
            if int(tmplist[2]) != -1:
                arc_2.nodes.append(node)
                arc_2.nodes = list(set(arc_2.nodes))

            #self.arcs.append(arc_1)

            key = self.make_arc_id(arc_1)
            arc_1.key = key

            arc_1.ogKey = key
            arc_1.ogIndex = arc_1.index
            arc_1.ogNodeIds = arc_1.node_ids

            self.arc_dict[key] = arc_1
            if arc_2 is not None:
                key = self.make_arc_id(arc_2)
                arc_2.key = key

                arc_2.ogKey = key
                arc_2.ogIndex = arc_2.index
                arc_2.ogNodeIds = arc_2.node_ids

                self.arc_dict[key] = arc_2
                #self.arcs.append(arc_2)
        self.arcs = list(self.arc_dict.values())

    def read_labels_from_file(self, file = None):
        label_file = open(file, "r")
        label_lines = label_file.readlines()
        label_file.close()
        arcs = []
        #print(self.gid_map.keys())
        for gid, l in enumerate(label_lines):
            tmplist = l.split(' ')
            label = 1. if int(tmplist[0]) == 1 else 0.
            arc = self.gid_map[gid]
            #arc = self.arc_dict[arc.key]
            arc.ground_truth = label
            arc.label_accuracy = label
            self.arc_dict[arc.key] = arc
            #arcs.append(arc)
        self.arcs = list(self.arc_dict.values())

    def build_node_map(self):
        self.node_map = {}
        current_arc_index = 0
        for arc in self.arcs:
            for node_id in arc.node_ids:
                if node_id not in self.node_map:
                    self.node_map[node_id] = []
                self.node_map[node_id].append(current_arc_index)
            current_arc_index += 1
        return self.node_map

    def build_select_node_map(self, arcs):
        self.select_node_map = {}
        current_arc_index = 0
        for arc in arcs:
            for node_id in arc.node_ids:
                if node_id not in self.select_node_map:
                    self.select_node_map[node_id] = []
                self.select_node_map[node_id].append(current_arc_index)
            current_arc_index += 1
        return self.select_node_map

    def build_kdtree(self):
        arc_points = []
        def make_arc_id(a):
            return tuple(a.node_ids) + (len(a.line),)
        for arc in sorted(self.arcs, key=lambda arc: len(arc.line)):
            index = make_arc_id(arc)
            arc_points.extend(arc.line)
            self.arc_map.extend([index] * len(arc.line))
        # only needed for selection ui to choose neighboring arcs
        # can cause error with sparse MSC
        self.kdtree = scipy.spatial.KDTree(arc_points, leafsize=10000)
        return self.kdtree

    def build_select_kdtree(self, arcs):
        arc_points = []
        def make_arc_id(a):
            return tuple(a.node_ids) + (len(a.line),)
        for arc in sorted(arcs, key=lambda arc: len(arc.line)):
            index = make_arc_id(arc)
            arc_points.extend(arc.line)
            self.selected_arc_map.extend([index] * len(arc.line))
        # only needed for selection ui to choose neighboring arcs
        # can cause error with sparse MSC
        self.select_kdtree = scipy.spatial.KDTree(arc_points, leafsize=10000)
        return self.select_kdtree

    def get_closest_arc_index(self, point):
        distance, index = self.kdtree.query(point)
        return self.arc_map[index]

    def get_closest_selected_arc_index(self, point):
        distance, index = self.select_kdtree.query(point)
        return self.selected_arc_map[index]

    def write_msc(self, filename, msc=None,  label=False):


        self.msc_edge_file = filename + ".arcs.txt"
        self.msc_vertex_file = filename + ".nodes.txt"
        # write line file
        def write_edge_file(arcs):
            edge_file = open(self.msc_edge_file,"w+")
            for arc in arcs:
                edge_file.write(str(arc.index)+",")
                for id in arc.node_ids:
                    edge_file.write(str(id)+",")
                if label:
                    edge_file.write(str(arc.label_accuracy)+",")
                for idx,point in enumerate(arc.line):
                    if idx != len(arc.line)-1:
                        edge_file.write(str(point[0])+","+str(point[1])+",")
                    else:
                        edge_file.write(str(point[0]) + "," + str(point[1])+"\n")

        # write node file
        def write_node_file(nodes):#line):
            vertex_file = open(self.msc_vertex_file, "w+")
            #tmplist = line.split(",")
            for node in nodes:
                # cellid
                vertex_file.write(str(node)+",")
                vertex_file.write(str(nodes[node].xy[0])+","+str(nodes[node].xy[1])+"\n")
        if msc is None:
            print("writing msc edge file: ")
            print(self.msc_edge_file)
            write_edge_file(self.arcs)
            write_node_file(self.nodes)
        else:
            write_edge_file(msc.arcs)
            write_node_file(msc.nodes)

    def write_arc_predictions(self, filename, msc=None):


        self.msc_pred_file = filename + ".preds.txt"
        pred_file = open(self.msc_pred_file,"w+")
        for arc in self.arcs:
            pred_file.write(str(arc.index)+",")
            pred_file.write(str(arc.prediction) + "\n")


    def draw_segmentation(self,  X, Y, filename, ridge=True, valley=True
                          , msc=None, invert=False, reshape_out=False, dpi=True
                          , original_image=None, type=None):

        #if msc is not None:
        #    self.msc = msc
        self.use_ridge_arcs = ridge
        self.use_valley_arcs = valley

        def make_arc_id(a):
            return tuple(a.node_ids) + (len(a.line),)

        black_box = np.zeros((X, Y)) if not invert else np.zeros(
            (Y, X))
        # print(self.image.shape+(,,3))
        #cmap = cm.get_cmap('Spectral')
        cmap = cm.get_cmap('bwr')
        cmap.set_under('black')
        cmap.set_bad('black')
        #cmap.set_over('white')
        plt.set_cmap(cmap)
        fig = plt.imshow(black_box, cmap=cmap, alpha=None, vmin=0)#, interpolation='nearest')  # plt.figure() #in
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        def is_ridge_arc(arc, nodes):
            return False# 0 not in [ nodes[arc.node_ids[0]].index, nodes[arc.node_ids[1]].index,]

        def is_valley_arc(arc, nodes):
            return False #2 not in [ nodes[arc.node_ids[0]].index, nodes[arc.node_ids[1]].index,]
        #print("image shape OG >>>> ")
        #print(original_image.shape)
        if original_image.shape[0] == 3:
            mapped_image = np.transpose(original_image, (2, 1, 0))#original_image.shape[2]))
        elif original_image.shape[1] == 3:
            mapped_image = np.transpose(original_image, (0, 2, 1))
        else:
            mapped_image = original_image
        mapped_image *= 255



        label_map_image = copy.deepcopy(mapped_image)

        for arc in self.arcs:#msc.arcs:
            if type is None:
                percent_overlap_color = cmap(arc.label_accuracy)
            #elif type=='partitions':
            #    label_color = cmap(0.1) if arc.partition=="train" else cmap(0.4) if arc.partition=="val" else cmap(0.9)
            else:
                if isinstance(arc.prediction,
                              (int, np.integer)) or isinstance(arc.prediction, (float, np.float)):
                    label_color = cmap(float(arc.prediction))
                else:
                    if len(arc.prediction) == 3:
                        label_color = cmap(0.5) if float(arc.prediction[2]) > 0.5 else cmap(float(arc.prediction[1]))
                    else:
                        label_color = cmap(float(arc.prediction[1]))

            if original_image is not None:
                x = 1 if invert else 0
                y = 0 if invert else 1
                if len(mapped_image.shape) == 2:
                    #arc.label_accuracy = 0.6
                    for p in np.array(arc.line):
                        mapped_image[int(p[x]), int(p[y])] = int(label_color[0] * 255)
                        mapped_image[int(p[x]), int(p[y])] = int(label_color[1] * 255)
                        mapped_image[int(p[x]), int(p[y])] = int(label_color[2] * 255)

                        msc_ground_seg_color = cmap(arc.label_accuracy)
                        label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[0] * 255)
                        label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[1] * 255)
                        label_map_image[int(p[x]), int(p[y])] = int(msc_ground_seg_color[2] * 255)
                else:
                    for p in np.array(arc.line):
                        mapped_image[int(p[x]), int(p[y]), 0] = int(label_color[0]*255)
                        mapped_image[int(p[x]), int(p[y]), 1] = int(label_color[1]*255)
                        mapped_image[int(p[x]), int(p[y]), 2] = int(label_color[2]*255)

                        msc_ground_seg_color = cmap(arc.label_accuracy)
                        label_map_image[int(p[x]), int(p[y]), 0] = int(msc_ground_seg_color[0] * 255)
                        label_map_image[int(p[x]), int(p[y]), 1] = int(msc_ground_seg_color[1] * 255)
                        label_map_image[int(p[x]), int(p[y]), 2] = int(msc_ground_seg_color[2] * 255)
            #print(label_color, " COLOR")

        # plt.gca().set_axis_off()
        # plt.gca().set_xlim(0, self.image.shape[0])
        # plt.gca().set_ylim(self.image.shape[1], 0)
        dpi_ = None
        if dpi:
            if X >= 600:
                dpi_ = 600
            else:
                dpi_ = 156
            if isinstance(dpi, int):
                dpi_ = dpi

        plt.savefig(filename, dpi=dpi_, cmap=cmap, bbox_inches='tight', pad_inches=0.001, transparent=False)

        img = imageio.imread(filename)

        if reshape_out and False:
            if X >= 600:
                img = Image.fromarray(img).resize((img.shape[1] + 9, img.shape[1] - 91 + 5))
            else:
                img = Image.fromarray(img).resize((img.shape[1] + 1, img.shape[0] + 1))
        else:
            img = img
        if original_image is not None:
            if len(mapped_image.shape) != 2:
                map_im = np.transpose(mapped_image, (0, 1, 2))
                lab_map_im = np.transpose(label_map_image, (0, 1, 2))
            else:
                map_im =mapped_image
                lab_map_im = label_map_image
            Img = Image.fromarray(map_im.astype('uint8'))  # .astype(np.float32))#mapped_img)
            Img.save(filename + 'MAP.' + filename.split('.')[-1])

            Img = Image.fromarray(
                lab_map_im.astype('uint8'))  # .astype(np.float32))#mapped_img)
            Img.save(filename + 'MAP_groundseg.' + filename.split('.')[-1])

        plt.close()

    def equate_graph(self, G):
        for arc, node in zip(self.arcs, G.nodes_iter()):
            points = np.array(np.round(arc.line), dtype=int)

            arc.label_accuracy = G.node[node]["label_accuracy"]
            if G.node[node]['train']:
                arc.partition = 'train'
            if G.node[node]['val']:
                arc.partition = 'val'
            if G.node[node]['test']:
                arc.partition = 'test'
            if G.node[node]["prediction"] is not None:
                arc.prediction = G.node[node]["prediction"]

                if isinstance(arc.prediction,
                                  (int, np.integer)) or isinstance(arc.prediction, (float, np.float)):
                    # if len(arc.prediction) == 3:
                    #    pred = cmap(0.5) if float(arc.prediction[2]) > 0.5 else cmap(float(arc.prediction[1]))
                    # else:
                    arc.prediction = float(arc.prediction)
                else:
                    # print("pred ", arc.prediction)
                    arc.prediction = float(arc.prediction[1])
            self.arc_dict[arc.key] = arc


class GeoMSC_Union(GeoMSC):
    def __init__(self, geomsc1=None, geomsc2=None, z=1):
        #GeoMSC.__init__(self)

        self.inference_msc_count = 0
        self.training_msc_count = 0

        self.max_arc_index = -1
        self.max_node_id = -1

        super().__init__() #GeoMSC_Union, self


        self.union_key_dict = {}

        self.z = 0
        self.test_geomsc = geomsc2

        self.geomsc_dict = {}
        self.geomsc_dict[self.z] = geomsc1

        self.z += 1
        self.geomsc_dict[self.z] = geomsc2




        self.nodes1 = geomsc1.nodes if geomsc1 is not None else {}
        self.arcs1 = geomsc1.arcs if geomsc1 is not None else []

        self.geomsc1 = geomsc1.geomsc if geomsc1 is not None else None

        self.msc1 = geomsc1.geomsc if geomsc1 is not None else None
        self.arc_map1 = geomsc1.arc_map if geomsc1 is not None else {}
        self.arc_dict1 = geomsc1.arc_dict if geomsc1 is not None else {}
        self.arcs1_index = []




        self.nodes2 = geomsc2.nodes if geomsc1 is not None else {}
        self.arcs2 = geomsc2.arcs if geomsc1 is not None else []
        self.geomsc2 = geomsc2.geomsc if geomsc1 is not None else None
        self.msc2 = geomsc2.geomsc if geomsc1 is not None else None
        self.arc_map2 = geomsc2.arc_map if geomsc1 is not None else {}
        self.arc_dict2 = geomsc2.arc_dict if geomsc1 is not None else {}
        self.arcs2_index = []

        self.max_degree = 0

        if geomsc1 is not None:
            self._union_nodes()
            self._union_arcs()

    def U(self, geomsc2):

        print(" ")
        print(">>>>> current z of new union ", self.z)
        print(" ")

        # z component to be assigned to second graph (geomsc2) used for
        # identification, z=0 used in training, z=1 used in inference

        geomsc1 = self.geomsc_dict[self.z]

        self.nodes1 = geomsc1.nodes
        self.arcs1 = geomsc1.arcs
        self.geomsc1 = geomsc1.geomsc

        self.msc1 = geomsc1.geomsc
        self.arc_map1 = geomsc1.arc_map
        self.arc_dict1 = geomsc1.arc_dict




        self.nodes2 = geomsc2.nodes
        self.arcs2 = geomsc2.arcs
        self.geomsc2 = geomsc2.geomsc
        self.msc2 = geomsc2.geomsc
        self.arc_map2 = geomsc2.arc_map
        self.arc_dict2 = geomsc2.arc_dict


        self.z += 1
        self.geomsc_dict[self.z] = geomsc2



        self.arcs2_index = []




        self._union_nodes()
        self._union_arcs()
    

    def get_unioned_geomscs(self):
        return self.geomsc

    def get_geomscs(self):
        self.invert_map()
        return (self.geomsc1, self.geomsc2, self.test_geomsc)

    def get_test_geomsc(self):
        return self.test_geomsc

    def get_unioned_arc_dict(self):
        return self.arc_dict

    def get_arc_dicts(self):
        self.invert_map()
        return (self.arc_dict1, self.arc_dict2)

    def get_unioned_arcs(self):
        return self.arcs

    def get_arcs(self):
        self.invert_map()
        return (self.arcs1, self.arcs2)

    def invert_map(self):

        geomsc1_arcs = []
        inference_geomsc_arcs = []

        inference_geomsc = self.geomsc_dict[1]
        self.test_geomsc = inference_geomsc
        collect_msc = False
        if self.test_geomsc is None:
            collect_msc = True
            self.test_geomsc = GeoMSC()


        for a in self.arcs:
            #a = self.arc_dict[a_key]

            #if False:#a.z == 0 and self.multi_union != 1:
            #    #a1_id = self.arc_union_dict[a_key]
            #arc1 = self.arc_dict1[a1_id]
            # map changes for unioned msc
            #arc1.label_accuracy = a.label_accuracy
            #arc1.prediction = a.prediction
            #key = a.key
            #self.geomsc1.arc_dict[key] = a
            #geomsc1_arcs.append(a)

            if a.z == 1 and a.intersection_arc != 1:
            
                #a2_id = self.arc_union_dict[a_key]
                #arc2 = self.arc_dict2[a2_id]
                # map changes for unioned msc
                #arc2.label_accuracy = a.label_accuracy
                #rc2.prediction = a.prediction
                key = a.ogKey #- self.union_key_dict[1][1]
                index = a.ogIndex
                nodeIDs = a.ogNodeIds

                a.key = key
                a.node_ids = nodeIDs
                a.index = index

                if collect_msc:
                    self.test_geomsc.arc_dict[a.ogKey] = a
                    inference_geomsc_arcs.append(a)
                else:
                    ogArc = self.test_geomsc.arc_dict[a.ogKey]
                    ogArc.prediction = a.prediction
                    #ogArc.label_accuracy = a.label_accuracy
                    self.test_geomsc.arc_dict[a.ogKey] = ogArc
                    inference_geomsc_arcs.append(ogArc)

        #if self.multi_union != 1:
        #    self.geomsc1.arcs = geomsc1_arcs
        self.test_geomsc.arcs = inference_geomsc_arcs
        #if self.multi_union != 1:
        #    self.msc1 = self.geomsc1
        self.msc2 = self.test_geomsc

        #self.test_geomsc = self.geomsc2

    def make_arc_id(self, a):
        return tuple(a.node_ids) + (len(a.line),)

    def __group_xy(self, lst):
        for i in range(0, len(lst), 2):
            yield tuple(lst[i : i + 2])

    def make_line(self, line, labeled=False):
        arc = [
                i for i in self.__group_xy([float(i) for i in line])
            ]  # read the rest of the the points in the arc as xy tuples
        return arc

    def connect_union(self):
        for arc_key in self.arc_dict1.keys():
            arc = self.arc_dict1[arc_key]

        self.max_arc_index = self.max_arc_index + 1
        self.max_node_id = self.max_node_id + 1

        a = MSCArc()
        a.index = self.max_arc_index
        a.label_accuracy = 1e-10
        a.intersection_arc = 1
        a.z=-1

        n0 = MSCNode()
        n1 = MSCNode()
        n0.cellid = self.max_node_id
        n1.cellid = self.max_node_id + 1
        n0.z = -1
        n1.z = -1
        n0.intersection_node = 1
        n1.intersection_node = 1

        self.max_node_id = self.max_node_id + 1

        recent_geomsc_key = self.z - 1

        g1 = self.geomsc_dict[recent_geomsc_key]

        g1_node_offset = self.union_key_dict[self.z - 1][0]
        g1_arc_offset = self.union_key_dict[self.z - 1][1]

        g2 = self.geomsc_dict[self.z]

        g2_node_offset = self.union_key_dict[self.z][0]
        g2_arc_offset = self.union_key_dict[self.z][1]

        arc1 = self.arcs[g1_arc_offset]
        arc1_line = arc1.line
        node1 = self.nodes[ arc1.node_ids[0] ]

        n0.xy = node1.xy
        n0.index = a.index# arc1.index + self.max_node_id_1
        n0.add_arc(a)

        if n0.degree > self.max_degree:
            self.max_degree = n0.degree

        arc2 = self.arcs[g2_arc_offset]
        node2 = self.nodes[ arc2.node_ids[0]] #arc2.node_ids[0] + self.max_node_id_2]
        i = 1
        while (node1.xy[0] - node2.xy[0]) == 0:
            arc2 = self.arcs[g2_arc_offset + i]
            node2 = self.nodes[arc2.node_ids[0] ]
            i+=1

        arc2_line = arc2.line

        a.node_ids = [arc1.node_ids[0] , arc2.node_ids[0] ]

        n1.index = a.index
        n1.xy = node2.xy
        n1.add_arc(a)

        if n1.degree > self.max_degree:
            self.max_degree = n1.degree

        euclidian_dist = np.sqrt((n0.xy[0] - n1.xy[0])**2 + (n0.xy[1] - n1.xy[1])**2 )
        p1 = np.sqrt((n0.xy[0] + n0.xy[1])**2 )
        p2 = np.sqrt((n1.xy[0] + n1.xy[1])**2 )

        m = (n0.xy[1] - n1.xy[1]) / (n0.xy[0] - n1.xy[0])
        b = -1 * m * n0.xy[0] + n0.xy[1]
        def __line(x):
            return m*x + b

        def __step(x,i,direction):
            if direction==-1:
                return x - i
            if direction==1:
                return x+i

        if n0.xy[0] > n1.xy[0]:
            dir = -1
        else:
            dir = 1

        a_line = [ tuple([ n0.xy[0] + dir*i ,  __line(n0.xy[0]) + dir*i]) for i in range(int(n0.xy[0] - dir*n1.xy[0]))]#euclidian_dist))]#

        a_line.append(tuple([ n1.xy[0],  __line(n1.xy[0]) ]) )

        a.line = [tuple([ n0.xy[0], n0.xy[1] ]) ] #a_line #self.make_line(a_line)

        min_line = (arc1,arc2) if len(np.array(arc1).line.flatten) < len(np.array(arc2).line.flatten) else (arc2, arc1)

        line = np.array(min_line[0]).flatten()  - np.array(np.array(min_line[1])).flatten()[0:len(np.array(min_line[0]))-1]

        for i in range(2,len(arc2.line)):
            a.line.append([tuple([ 1,1]) ])

        a.line.append([tuple([ n1.xy[0], n1.xy[1] ]) ])

        a.nodes = [n0, n1]
        self.arcs.append(a)
        a_id = self.make_arc_id(a)
        a.key = a_id

        self.arc_dict[a_id] = a




        #self.arc_union_dict[a_id] = -1

        self.nodes[n0.cellid] = n0
        self.nodes[n1.cellid] = n1


        #


    def _union_nodes(self):

        self.max_node_id_1 = self.max_node_id + 1
        self.node_offset_1 = self.max_node_id_1

        self.max_node_id_2 = self.max_node_id

        for n1 in self.nodes1.keys():
            node = self.nodes1[n1]
            n = MSCNode()

            n.cellid = node.cellid + self.node_offset_1

            self.max_node_id_1 = self.max_node_id_1 + 1
            self.max_node_id_2 = n.cellid + 1

            #if n.cellid > self.max_node_id:
            #    self.max_node_id = n.cellid

            n.index = None
            n.xy = node.xy

            # either first (0) msc or last added to geomsc dict
            n.z = self.z - 1

            self.nodes[n.cellid] = n


        self.node_offset_2 = self.max_node_id_2

        for n2 in self.nodes2.keys():
            node = self.nodes2[n2]
            n = MSCNode()
            n.cellid = node.cellid + self.node_offset_2#self.nodeid_switch_key

            self.max_node_id_2 = n.cellid

            n.index = None
            n.xy = node.xy

            # switch component to id which msc came from
            n.z = self.z

            self.nodes[n.cellid] = n

        self.max_node_id = self.max_node_id_2



        #self.nodes = self.geomsc.nodes

    def _union_arcs(self):

        self.max_arc_index_1 = self.max_arc_index + 1
        self.arc_offset_1 = self.max_arc_index_1

        self.max_arc_index_2 = self.max_arc_index_1

        self.union_key_dict[self.z - 1] = [self.node_offset_1 , self.arc_offset_1]

        def make_arc_id(a):
            return tuple(a.node_ids) + (len(a.line),)


        for arc_key in self.arcs1:#self.arc_dict1.keys():
            arc = arc_key # self.arc_dict1[arc_key]
            a = MSCArc()

            a.index = arc.index + self.arc_offset_1

            self.max_arc_index_1  = self.max_arc_index_1 + 1
            self.max_arc_index_2 = a.index + 1

            a.label_accuracy = arc.label_accuracy

            a.node_ids = [ nid + self.node_offset_1 for nid in arc.node_ids] #arc.node_ids

            a.line = arc.line



            n1 = self.nodes[a.node_ids[0] ]
            n2 = self.nodes[a.node_ids[1] ]
            n1.index = a.index
            n2.index = a.index
            n1.add_arc(a)
            n2.add_arc(a)

            if n1.degree > self.max_degree:
                self.max_degree = n1.degree
            if n2.degree > self.max_degree:
                self.max_degree = n2.degree

            a.nodes = [n1, n2]
            self.arcs.append(a)
            a_id = make_arc_id(a)

            a.key = a_id

            self.arc_dict[a_id] = a

            a.ogKey = arc.ogKey
            a.ogIndex = arc.ogIndex
            a.ogNodeIds = arc.ogNodeIds

            a.z = self.z - 1


        self.arc_offset_2 = self.max_arc_index_2
        self.union_key_dict[self.z ] = [self.node_offset_2, self.arc_offset_2]

        for arc_key in self.arcs2: #_dict2.keys():
            arc = arc_key#self.arc_dict2[arc_key]
            a = MSCArc()

            a.z = self.z

            a.index = arc.index + self.arc_offset_2
            self.max_arc_index_2 = a.index

            a.label_accuracy = arc.label_accuracy
            a.node_ids = [ nid + self.node_offset_2 for nid in arc.node_ids]
            a.line = arc.line
            n1 = self.nodes[a.node_ids[0]  ]
            n2 = self.nodes[a.node_ids[1]  ]
            n1.index = a.index
            n2.index = a.index
            n1.add_arc(a)
            n2.add_arc(a)
            a.nodes = [n1, n2]
            self.arcs.append(a)
            a_id = make_arc_id(a)
            a.key = a_id

            a.ogKey = arc.ogKey# _key
            a.ogIndex = arc.ogIndex
            a.ogNodeIds = arc.ogNodeIds

            self.arc_dict[a_id] = a

        self.max_arc_index = self.max_arc_index_2


        self.geomsc = self
        #self.geomsc.arcs = self.arcs
        # self.geomsc.arc_dict = self.arc_dict

    def node_read_from_line(self, line, node):
        tmplist = line.split(",")
        node.cellid = int(tmplist[0])
        node.index = None #int(tmplist[1])
        #self.value = float(tmplist[2])
        #self.boundary = int(tmplist[3])
        node.xy = (float(tmplist[1]), float(tmplist[2]))
        node.z = int(tmplist[3])
        return node

    def arc_read_from_line(self, line, arc, labeled=False):
        tmplist = line.split(",")
        arc.index = int(tmplist[0])
        if labeled:
            arc.label_accuracy = float(tmplist[3])
            arc.node_ids = [int(tmplist[1]), int(tmplist[2])]

            arc.ogKey = tuple([int(tmplist[4]), int(tmplist[5])]) + (int(tmplist[6]),)
            arc.ogIndex = int(tmplist[7])
            arc.ogNodeIds = [int(tmplist[8]), int(tmplist[9])]
            arc.z = int(tmplist[10])

            #arc.intersection_arc = int(tmplist[11])
            arc.line = [
                i for i in self.__group_xy([float(i) for i in tmplist[11:]])
            ]  # read the rest of the the points in the arc as xy tuples
        else:
            arc.node_ids = [int(tmplist[1]), int(tmplist[2])]
            arc.ogKey = tuple([int(tmplist[3]), int(tmplist[4])]) + (int(tmplist[5]),)
            arc.ogIndex = int(tmplist[6])
            arc.ogNodeIds = [int(tmplist[7]), int(tmplist[8])]
            arc.z = int(tmplist[9])
            #arc.intersection_arc = int(tmplist[10])
            arc.line = [
                i for i in self.__group_xy([float(i) for i in tmplist[10:]])
            ]#read the rest of the the points in the arc as xy tuples
        return arc

    def read_from_file(self, fname_base, labeled=False):
        nodesname = fname_base + ".nodes.txt"
        arcsname = fname_base + ".arcs.txt"
        node_file = open(nodesname, "r")
        nodes_lines = node_file.readlines()
        node_file.close()
        for l in nodes_lines:
            n = MSCNode()
            n = self.node_read_from_line(l, n)
            self.nodes[n.cellid] = n
        arcs_file = open(arcsname, "r")
        arcs_lines = arcs_file.readlines()
        arcs_file.close()
        for l in arcs_lines:
            a = MSCArc()
            a = self.arc_read_from_line(l, a, labeled)

            a.label_accuracy = None
            a.partition = None
            a.prediction = None

            n1 = self.nodes[a.node_ids[0]]
            n2 = self.nodes[a.node_ids[1]]
            n1.index = a.index
            n2.index = a.index
            n1.add_arc(a)
            n2.add_arc(a)
            a.nodes = [n1, n2]
            key = self.make_arc_id(a)
            a.key = key

            a.ogKey = key
            a.ogIndex = a.index
            a.ogNodeIds = a.node_ids


            self.arcs.append(a)
            self.arc_dict[key] = a


    def write_msc(self, filename, msc=None,  label=False):


        self.msc_edge_file = filename + ".arcs.txt"
        self.msc_vertex_file = filename + ".nodes.txt"
        # write line file
        def write_edge_file(arcs):
            edge_file = open(self.msc_edge_file,"w+")
            for arc in arcs:
                edge_file.write(str(arc.index)+",")
                for id in arc.node_ids:
                    edge_file.write(str(id)+",")
                if label:
                    edge_file.write(str(arc.label_accuracy)+",")
                for k in arc.ogKey:
                    edge_file.write(str(k)+",")
                edge_file.write(str(arc.ogIndex) + ",")
                for og_id in arc.ogNodeIds:
                    edge_file.write(str(og_id) + ",")
                edge_file.write(str(arc.z)+",")
                #edge_file.write(str(arc.intersection_arc)+",")
                for idx,point in enumerate(arc.line):
                    if idx != len(arc.line)-1:
                        edge_file.write(str(point[0])+","+str(point[1])+",")
                    else:
                        edge_file.write(str(point[0]) + "," + str(point[1])+"\n")

        # write node file
        def write_node_file(nodes):#line):
            vertex_file = open(self.msc_vertex_file, "w+")
            #tmplist = line.split(",")
            for node in nodes:
                # cellid
                vertex_file.write(str(node)+",")
                vertex_file.write(str(nodes[node].xy[0])+","+str(nodes[node].xy[1])+",")
                vertex_file.write(str(nodes[node].z) + "\n")
        if msc is None:
            print("writing msc edge file: ")
            print(self.msc_edge_file)
            write_edge_file(self.arcs)
            write_node_file(self.nodes)
        else:
            write_edge_file(msc.arcs)
            write_node_file(msc.nodes)