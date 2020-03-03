# Standard library imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy

# Third party imports
import imageio
from PIL import Image

# Local application
#from topoml.topology.utils import is_ridge_arc, is_valley_arc
from topoml.image.utils import make_mc_arc_mask

#print("%%%%%%%%%%%%%%%%%%%%%% ", os.getcwd())

class MSCNode:
    def __init__(self):
        self.arcs = []

    def read_from_line(self, line):
        tmplist = line.split(",")
        self.cellid = int(tmplist[0])
        self.index = None #int(tmplist[1])
        #self.value = float(tmplist[2])
        #self.boundary = int(tmplist[3])
        self.xy = (float(tmplist[1]), float(tmplist[2]))

    def add_arc(self, arc):
        self.arcs.append(arc)


class MSCArc:
    def __init(self):
        self.nodes = []
        self.line = []
        self.label_accuracy = None

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
        self.arc_dict = {}


    def make_arc_id(self,a):
        return tuple(a.node_ids) + (len(a.line),)

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
            a.nodes = [n1, n2]
            self.arcs.append(a)
            self.arc_dict[self.make_arc_id(a)] = a

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
        self.kdtree = scipy.spatial.KDTree(arc_points, leafsize=1000)
        return self.kdtree

    def get_closest_arc_index(self, point):
        distance, index = self.kdtree.query(point)
        return self.arc_map[index]

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

    def draw_segmentation(self,  X, Y, filename, ridge=True, valley=True
                          , msc=None, invert=False, reshape_out=False, dpi=True, original_image=None):

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

        if original_image.shape[0] == 3:
            mapped_image = np.transpose(original_image, (1, 2, 0))#original_image.shape[2]))
        elif original_image.shape[1] == 3:
            mapped_image = np.transpose(original_image, (2, 0, 1))
        else:
            mapped_image = original_image
        mapped_image *= 255

        for arc in self.msc.arcs:
            label_color = cmap(arc.label_accuracy)
            if original_image is not None:
                x = 0 if invert else 1
                y = 1 if invert else 0
                for p in np.array(arc.line):
                    mapped_image[int(p[x]), int(p[y]), 0] = int(label_color[0]*255)
                    mapped_image[int(p[x]), int(p[y]), 1] = int(label_color[1]*255)
                    mapped_image[int(p[x]), int(p[y]), 2] = int(label_color[2]*255)
            #print(label_color, " COLOR")
            """
            if (not self.use_ridge_arcs and is_ridge_arc(arc, self.nodes)) or (
                    not self.use_valley_arcs and is_valley_arc(arc, self.nodes)
            ):
                continue
            
            arc_index = make_arc_id(arc)
            if arc_index in self.in_arcs:
                points = np.array(arc.line)
                self.arc_drawings[arc_index] = plt.scatter(
                    points[:, 0],
                    points[:, 1],
                    facecolor=label_color,#'white',  # self.in_color,
                    alpha=None,
                    edgecolor="none",
                    s=5,
                    marker=",",
                    zorder=3,
                )
                self.arc_drawings[arc_index].set_visible(True)
            elif arc_index in self.out_arcs:
                points = np.array(arc.line)
                self.arc_drawings[arc_index] = plt.scatter(
                    points[:, 0],
                    points[:, 1],
                    facecolor=label_color,#'white',  # self.out_color,
                    edgecolor="none",
                    alpha=None,
                    s=5,
                    marker=",",
                    zorder=3,
                )
                self.arc_drawings[arc_index].set_visible(True)"""

        if self.use_ridge_arcs:
            arc_mask = make_mc_arc_mask(None, self, X, Y, invert)  # False)
            plt.imshow(
                arc_mask,
                cmap=cmap,
                vmin=0,
                #vmax=0,
                #norm=plt.Normalize(vmin=0, vmax=1),
                #interpolation="nearest",
                alpha=None,
                zorder=2,
            )
        if self.use_valley_arcs:
            arc_mask = make_mc_arc_mask(None, self, X, Y, invert)  # True)
            plt.imshow(
                arc_mask,
                cmap=cmap,#"binary",
                vmin=0,
                #vmax=0,
                #interpolation="nearest",
                alpha=None,
                zorder=2,
            )

        """
        extrema_points = [[], [], []]
        for node in self.msc.nodes.values():
            x, y = node.xy
            extrema_points[node.index].append([x, y])

        for i, color in enumerate([blue, green, red]):
            xy = np.array(extrema_points[i])
            plt.scatter(
                xy[:, 0],
                xy[:, 1],
                facecolor=color,
                edgecolor="none",
                s=1,
                marker=",",
                zorder=4,
            )
        """

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
            Img = Image.fromarray(np.transpose(mapped_image, (0, 1, 2)).astype('uint8'))  # .astype(np.float32))#mapped_img)
            Img.save(filename + 'MAP.' + filename.split('.')[-1])
            """original_image = np.transpose(original_image, (1, 2, 0))
            zeros = np.zeros((original_image.shape[0], original_image.shape[1], 3))#original_image.shape[2]))
            x =np.array(img).shape[0]
            y = np.array(img).shape[1]
            zeros[:x,:y,:] = np.array(img)[:x,:y,:3]
            img = zeros
            x = np.array(img).shape[0]
            y = np.array(img).shape[1]
            #print(original_image.min(), "min")
            #print(original_image.max(), "max")
            img_mask = np.array(img).copy()
            #np.array(img_mask)[np.array(img, dtype=np.int32) == 0] = -1
            #np.array(img_mask)[np.array(img_mask) > 0] = 0
            original_image = original_image - img
            original_image[original_image<0] = 0
            original_image*=255
            mapped_img = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3))#,np.float32)
            mapped_img[:,:,0] = np.array(img)[:x,:y,0]+ np.array(original_image)[:x,:y,0]#/255.#- np.array(img)[:x,:y,0]/255.)#/255.#[:, :, 0]
            mapped_img[:, :, 1] = np.array(img)[:x, :y, 1] + np.array(original_image)[:x, :y,1]#/255.  # [:, :, 0]
            mapped_img[:, :, 2] = np.array(img)[:x, :y, 2] + np.array(original_image)[:x, :y,2]#/255.  # [:, :, 0]
            #print("im shape msc ", np.array(img).shape, " pg img ", np.array(original_image).shape)
            #print("mapped im shape ", mapped_img.shape)
            #mapped_img *= 255
            #mapped_img[mapped_img > 255] = 255
            Img = Image.fromarray(np.transpose(mapped_img, (0,1, 2)).astype('uint8'))#.astype(np.float32))#mapped_img)
            Img.save(filename+'MAP.'+filename.split('.')[-1])
            """
        plt.close()
        """
        in_arcs = []
        out_arcs = []
        for arc in self.msc.arcs:
            index = make_arc_id(arc)
            if index in self.in_arcs:
                in_arcs.append(arc)
            elif index in self.out_arcs:
                out_arcs.append(arc)

        return (in_arcs, out_arcs, np.array(list(self.out_pixels)))"""
