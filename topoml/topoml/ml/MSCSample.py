#3rd party imports
import numpy as np
import networkx as nx
import samply
import scipy.stats
import json
from networkx.readwrite import json_graph

#local imports
from topoml.ui.ArcSelector import ArcSelector
from topoml.topology.utils import (
    get_pixel_values_from_arcs,
    is_ridge_arc,
    is_valley_arc,
)
from topoml.ml.utils import gaussian_fit

from topoml.image.utils import warp_polar


from topoml.image.feature import (
    mean_filter,
    variance_filter,
    median_filter,
    minimum_filter,
    maximum_filter,
    gaussian_blur_filter,
    difference_of_gaussians_filter,
    sobel_filter,
    cosine_similarity,
    hyperbolic_distance,
    laplacian_filter,
    neighbor_filter,
)
from topoml.topology.geometric_msc import GeoMSC


class MSCSample():
    def __init__(self, msc=None, geomsc=None, msc_collection=None, image=None
                 , features={}, labeled_segmentation=None, use_inference = None, ridge=True, valley=True, select_label=False):

        self.msc = msc if msc is not None else geomsc
        self.geomsc = geomsc if geomsc is not None else msc
        self.msc_collection = msc_collection if msc_collection is not None else {}
        self.arcs = None
        self.ridge = ridge
        self.valley= valley
        self.arc_map = None
        self.arc_accuracies = None
        self.labeled_segmentation = None

        self.select_label = select_label

        # msc after labeling arcs (updating arc.label_accuracy
        self.labeled_msc = None
        self.image = image
        self.features = features
        self.lineG = None
        if self.msc is not None:
            self.assign_msc(self.msc)

        self.use_inference = use_inference
        self.compiled_features = None

    def assign_msc(self, msc):
        self.msc = msc
        self.geomsc = msc
        self.arcs = msc.arcs
        self.arc_dict = msc.arc_dict
        self.arc_map = msc.arc_map
        self.nodes = msc.nodes


    def msc_arc_accuracy(self, arc=None, msc=None, geomsc=None, labeled_segmentation=None, labeled_mask=None, invert=True):
        if labeled_segmentation is not None:
            self.labeled_segmentation = labeled_segmentation
        arc_accuracy = 0
        percent_interior = 0
        for point in arc.line:
            x = 0
            y = 1
            if invert:
                x = 1
                y = 0
            if self.labeled_segmentation[int(point[x]),int(point[y])] > 0:
                arc_accuracy += 1.
            if labeled_mask[int(point[x]),int(point[y])] > 0:
                percent_interior += 1.
        label_accuracy = arc_accuracy/float(len(arc.line))
        interior_percent_arc = percent_interior/float(len(arc.line))
        if interior_percent_arc < .85:
            arc.exterior = 1
        if label_accuracy == 0.:
            label_accuracy = 1e-4
        return label_accuracy



    def map_labeling(self, image=None, msc=None, geomsc=None, labeled_segmentation=None, labeled_mask = None, invert=False):
        if msc is not None:
            self.msc = msc
        if geomsc is not None:
            self.geomsc = geomsc
            self.msc = geomsc
        if labeled_segmentation is not None:
            self.labeled_segmentation = labeled_segmentation

        # This ensures smaller arcs take precedence in the event of
        # pixel overlap
        """sorted_arcs = sorted(self.msc.arcs, key=lambda arc: len(arc.line))
        arc_points = []
        self.arcs = []
        for arc in sorted_arcs:
            if (not self.ridge and is_ridge_arc(arc, self.msc)) or (
                    not self.valley and is_valley_arc(arc, self.msc)
            ):
                continue
            index = tuple(arc.node_ids) + (len(arc.line),)#make_arc_id(arc)
            arc_points.extend(arc.line)
            self.arc_map.extend([index] * len(arc.line))
            self.arcs.append(arc)
        """

        for arc in self.msc.arcs:
            arc.label_accuracy = self.msc_arc_accuracy(arc=arc
                                                        , labeled_segmentation=self.labeled_segmentation
                                                        ,labeled_mask=labeled_mask
                                                        ,invert=invert)


        self.labeled_msc = self.msc
        return self.msc

    def sort_labeled_arcs(self, msc=None, geomsc=None, accuracy_threshold=0.1, bin=[]):
        if msc is not None:
            self.msc = msc
        if geomsc is not None:
            self.geomsc = geomsc
            self.msc = geomsc

        self.positive_arc_ids = set()
        self.negative_arc_ids = set()
        self.positive_arcs = set()
        self.negative_arcs = set()

        for arc in self.msc.arcs:
            if arc.label_accuracy >= 1.0-accuracy_threshold:
                self.positive_arc_ids.add(self.msc.make_arc_id(arc))
                self.positive_arcs.add(arc)
            else:
                self.negative_arc_ids.add(self.msc.make_arc_id(arc))
                self.negative_arcs.add(arc)
        print("positive arcs: ", len(self.positive_arc_ids))
        print("negative arcs: ", len(self.negative_arc_ids))
        return (self.positive_arc_ids, self.negative_arc_ids)

    def sample_graph_neighborhoods(self, X, Y, msc=None
                                   , accuracy_threshold=0.1, count=2, rings=5, seed=0
                                   , validation=False, test=False):
        np.random.seed(seed)
        hypercube = samply.hypercube.cvt(count, 2)
        hypercube[:, 0] *= X
        hypercube[:, 1] *= Y
        if msc is not None:
            self.msc = msc
        one_msc = self.msc#.geomsc_dict[0]

        all_selected_arcs = []
        if self.select_label:
            for i in self.selected_positive_arcs:
                all_selected_arcs.append(i)
            for i in self.selected_negative_arcs:
                all_selected_arcs.append(i)

        one_msc.build_kdtree() if not self.select_label else one_msc.build_select_kdtree(all_selected_arcs)#self.msc.build_kdtree()
        node_map = one_msc.build_node_map() #if not self.select_label else one_msc.build_select_node_map(all_selected_arcs)#self.msc.build_node_map()

        seed_arc_keys = list()

        #hypercube = tuple([arc.line[0] for arc in tuple(self.positive_arcs) + tuple(self.negative_arcs)]) \
        #    if self.select_label else hypercube

        for x in hypercube:
            print(" >>>> hypercube x", x)
            arc_key = one_msc.get_closest_arc_index(x) if not self.select_label else one_msc.get_closest_selected_arc_index(x) #self.msc.get_closest_arc_index(x)
            seed_arc_keys.append(arc_key)
        ring = 0
        ring_index = 0
        ring_count = len(seed_arc_keys)

        while ring <= rings:
            next_ring = seed_arc_keys[ring_index:(ring_index + ring_count)]

            ring_count = 0
            for arc_key in next_ring:
                for node_id in arc_key[:2]:
                    for arc_index in node_map[node_id]:
                        neighbor_arc_key = one_msc.make_arc_id(one_msc.arcs[arc_index])##self.msc.make_arc_id(self.msc.arcs[arc_index])
                        if neighbor_arc_key not in seed_arc_keys:
                            seed_arc_keys.append(neighbor_arc_key)
                            ring_count += 1
                ring_index += 1
            ring += 1
        print('size seed arc keys ', len(seed_arc_keys))
        seed_arc_keys = set(seed_arc_keys)

        """
        for arc_set in [self.in_arcs, self.out_arcs]:
            for arc_key in self.arc_map:
                if arc_key in arc_set and arc_key not in seed_arc_keys:
                    arc_set.remove(arc_key)
        """
        #if not self.select_label:
        self.sort_labeled_arcs(accuracy_threshold=accuracy_threshold)

        print("map output arc ")
        pos_arcs = self.positive_arc_ids if not self.select_label else self.selected_positive_arcs
        neg_arcs = self.negative_arc_ids if not self.select_label else self.selected_negative_arc_ids
        sample_set_ids = (self.positive_arc_ids.intersection(seed_arc_keys), self.negative_arc_ids.intersection(seed_arc_keys))
        subgraph_positive_arcs = set()
        subgraph_negative_arcs = set()
        for arc_id in sample_set_ids[0]:
            subgraph_positive_arcs.add(self.msc.arc_dict[arc_id])
        for arc_id in sample_set_ids[1]:
            subgraph_negative_arcs.add(self.msc.arc_dict[arc_id])
        if validation:
            self.validation_set["positive"] = subgraph_positive_arcs
            self.validation_set["negative"] = subgraph_negative_arcs
            self.validation_set_ids["positive"] = sample_set_ids[0]
            self.validation_set_ids["negative"] = sample_set_ids[1]
        if test:
            self.test_set["positive"] = subgraph_positive_arcs
            self.test_set["negative"] = subgraph_negative_arcs
            self.test_set_ids["positive"] = sample_set_ids[0]
            self.test_set_ids["negative"] = sample_set_ids[1]
        return subgraph_positive_arcs, subgraph_negative_arcs

    def create_graphsage_input(self, validation_positive_arcs=None, validation_negative_arcs=None
                               , test_positive_arcs=None, test_negative_arcs=None
                               , groundtruth_positive_arcs=None, groundtruth_negative_arcs=None
                               , label_scheme='train', val_count=20
                               , labeled_msc=True, msc=None):
        """ Creates the necessary input data structures for GraphSage's
            GNN implementation.

        Keyword arguments:
            validation_positive_arcs -- a set of arc ids representing validation set subgraph with
                                        edges accurate in segmentation up to some threshold in overlap
                                        with hand done groundtruth segmentation.
            validation_negative_arcs -- a set of arc ids representing validation set subgraph with
                                        edges in 'background', e.g. inaccurate in segmentation, up
                                        to some threshold
            background_indices -- a set of arcs representing
                                  "background" selection

        Returns:
            tuple: (string -- A networkx-specified json representation
                              describing the input graph. Nodes have
                              'val' and 'test' attributes specifying if
                              they are a part of the validation and test
                              sets, respectively.
                    string -- A json-stored dictionary mapping the graph
                              node ids to consecutive integers.
                    string -- A json-stored dictionary mapping the graph
                              node ids to classes.
                    ndarray -- An array of node features; ordering given
                               by second item. Can be omitted and only
                               identity features will be used.)
        """
        if msc is not None:
            self.assign_msc(msc)

        selected_indices = set()
        background_indices = set()
        gt_pos_indices = set()
        gt_neg_indices = set()
        test_pos_indices = set()
        test_neg_indices = set()

        if not labeled_msc:
            if not groundtruth_positive_arcs:
                groundtruth_positive_arcs = validation_positive_arcs
                groundtruth_negative_arcs = validation_negative_arcs
        else:
            self.sort_labeled_arcs()
            groundtruth_positive_arcs = self.positive_arc_ids
            groundtruth_negative_arcs = self.negative_arc_ids

        for arc in groundtruth_positive_arcs:
            gt_pos_indices.add(arc)
        for arc in groundtruth_negative_arcs:
            gt_neg_indices.add(arc)

        for arc in validation_positive_arcs:
            selected_indices.add(arc)
        for arc in validation_negative_arcs:
            background_indices.add(arc)
        val_set = validation_positive_arcs.union(validation_negative_arcs)

        for arc in test_positive_arcs:
            test_pos_indices.add(arc)
        for arc in test_negative_arcs:
            test_neg_indices.add(arc)
        test_set = test_pos_indices.union(test_neg_indices)

        if self.compiled_features is None:
            compiled_data = self.compile_features()
        else:
            compiled_data = self.compiled_features

        self.node_map = {}
        if self.lineG is None:
            G = self.construct_line_graph()
        else:
            G = self.lineG

        current_arc_id = 0
        node_ids = {}
        node_labels = {}

        i_val = 0  ##!!!! need to set size for val set

        for arc, features in zip(self.arcs, compiled_data):
            index = tuple(arc.node_ids) + (len(arc.line),)
            for node_id in arc.node_ids:

                if node_id not in self.node_map:
                    self.node_map[node_id] = []
                self.node_map[node_id].append(current_arc_id)

                # ensure edge of msc used is labeled.
                label = [
                    int(index in gt_neg_indices),
                    int(index in gt_pos_indices),
                ]

                node = G.node[current_arc_id]
                node["index"] = arc.node_ids
                node["size"] = len(arc.line)
                node["features"] = features.tolist()

                # labeled nodes assigned as train, test, or val
                if bool(np.sum(label)):
                    # label node as accurate or inaccurate in segmentation
                    # reserve key for inference
                    # assign key for observed accuracy based on percent
                    # overlap with hand done segmentations
                    node["label"] = label
                    node["prediction"] = None
                    node["label_accuracy"] = arc.label_accuracy
                    modified = 0
                    if index in val_set:
                        modified = 1
                        node["train"] = False
                        node["test"] = False
                        node["val"] = True
                    if index in test_set:
                        modified = 1
                        node["train"] = False
                        node["test"] = True
                        node["val"] = False
                    else:  # elif not bool(modified):
                        node["test"] = False
                        node["val"] = False
                        node["train"] = True

                """Label all non-selected arcs as test"""
                # if not  bool(np.sum(label)):
                # node["test"] = True
                # G.node[current_arc_id] = node

                # current_arc_id += 1
                node_ids[current_arc_id] = current_arc_id
                node_labels[current_arc_id] = label
            current_arc_id += 1

        for arc_id, arc in list(G.nodes_iter(data=True)):  # G.nodes.items():
            for node_id in arc["index"]:
                for connected_arc_id in self.node_map[node_id]:
                    G.add_edge(arc_id, connected_arc_id)

        data1 = json_graph.node_link_data(G)
        s1 = json.dumps(data1)  # input graph
        s2 = json.dumps(node_ids)  # dict: nodes to ints
        s3 = json.dumps(node_labels)  # dict: node_id to class

        return (data1, node_ids, node_labels, compiled_data)  # (s1, s2, s3, compiled_data)

    def construct_line_graph(self):
        """ Constructs a dual graph where Morse-Smale Arcs become nodes
            and edges represent when two Arcs share a critical point or
            otherwise intersect.
        """
        self.node_map = {}
        G = nx.Graph()
        current_arc_id = 0
        for arc in self.arcs:
            for node_id in arc.node_ids:
                if node_id not in self.node_map:
                    self.node_map[node_id] = []
                self.node_map[node_id].append(current_arc_id)
            G.add_node(
                current_arc_id, index=arc.node_ids, size=len(arc.line)
            )
            current_arc_id += 1

        for arc_id, arc in list(G.nodes_iter(data=True)):

            for node_id in arc["index"]:
                for connected_arc_id in self.node_map[node_id]:
                    if arc_id != connected_arc_id:
                        G.add_edge(arc_id, connected_arc_id)
        self.lineG = G
        return self.lineG.copy()

    def cosine_similarity(self, a1, a2):
        min_len = min(len(a1), len(a2))
        uv_x = np.transpose(np.array(a1)[:,0][0:min_len]).dot(np.array(a2)[:,0][0:min_len])
        uv_y = np.transpose(np.array(a1)[:,1][0:min_len]).dot(np.array(a2)[:,1][0:min_len])
        uv = uv_x + uv_y
        mag = np.linalg.norm(np.array(a1)) * np.linalg.norm(np.array(a2))
        cos_sim = uv/mag
        return float(cos_sim)

    def set_default_features(self, image=None, images=None,min_number_features=1, number_features = 5):
        if image is not None:
            self.image = image
        self.images = images if images is not None else {}

        # Functions to apply to the pixels of an arc
        self.features["length"] = lambda pixels: len(pixels)
        self.features["min"] = lambda pixels: np.min(pixels)
        self.features["max"] = lambda pixels: np.max(pixels)
        self.features["median"] = lambda pixels: np.median(pixels)
        #self.features["mode"] = lambda pixels: scipy.stats.mode(np.round(pixels, 2))[0][0]
        #self.features["mean"] = lambda pixels: gaussian_fit(pixels)[0]

        #self.features["std"] = lambda pixels: gaussian_fit(pixels)[1]
        self.features["var"] = lambda pixels: gaussian_fit(pixels)[2]

        self.features["skew"] = lambda pixels: scipy.stats.skew(pixels)
        self.features["kurtosis"] = lambda pixels: scipy.stats.kurtosis(pixels)
        self.features["range"] = lambda pixels: np.max(pixels) - np.min(pixels)

        # The input for these is fundamentally different, so for now
        # we will key off the fact that the name will always start with
        # "neighbor" in order to pass the right pixels to them.
        # Alternatively, we can send all of the pixels, and in the
        # methods above, just operate on the first row which could be
        # guaranteeed to be the pixels for the arc being operated on.
        """self.features["neighbor_degree"] = lambda connected_pixels: len(
            connected_pixels
        )
        self.features["neighbor_min"] = lambda connected_pixels: np.min(
            [np.min(pixels) for pixels in connected_pixels]
        )
        self.features["neighbor_max"] = lambda connected_pixels: np.max(
            [np.max(pixels) for pixels in connected_pixels]
        )
        self.features["neighbor_mean"] = lambda connected_pixels: np.mean(
            [np.mean(pixels) for pixels in connected_pixels]
        )
        self.features["neighbor_std"] = lambda connected_pixels: np.std(
            [np.mean(pixels) for pixels in connected_pixels]
        )"""

        # Pixel values to use for the aforementioned functions:
        self.images["identity"] = self.image

        #first derivatives
        print("sobel")
        self.images["sobel"] = sobel_filter(self.image)

        # second derivatives
        #print("lablacian")
        #self.images["laplacian"] = laplacian_filter(self.image)
        for i in range(min_number_features, number_features):
            pow1 = 2 ** (i - 1)
            pow2 = 2 ** i
            #print("mean: ", i)
            #self.images["mean_{}".format(i)] = mean_filter(self.image, i)
            print("variance: ", i)
            self.images["variance_{}".format(i)] = variance_filter(
                self.image, i
            )
            print("median: ",i)
            self.images["median_{}".format(i)] = median_filter(self.image, i)
            print("min/max: ", i)
            self.images["min_{}".format(i)] = minimum_filter(self.image, i)
            self.images["max_{}".format(i)] = maximum_filter(self.image, i)
            #print("gauss: ", i)
            #self.images["gauss_{}".format(pow2)] = gaussian_blur_filter(
            #    self.image, pow2
            #)

            #print("delta gauss: ",i)
            #self.images[
            #    "delta_gauss_{}_{}".format(pow1, pow2)
            #] = difference_of_gaussians_filter(self.image, pow1, pow2)

        print("map to polar")
        middle = lambda x: (0.5 * x[0], 0.5 * x[1])
        self.images["map_to_polar"] = warp_polar(self.image, center = middle(self.image.shape)
                                , radius = np.max(self.image.shape)/2.0
                                , output_shape = self.image.shape)
        #print("neighbor filter: ")
        #if number_features != 1:
        #    for i, neighbor_image in enumerate(neighbor_filter(self.image, min_shift=1, max_shift=3)):   # ___________pixel nbrhd  ____
        #        print(i)
        #        self.images["shift_{}".format(i)] = neighbor_image

        #self.kernel_stack = self.images["identity"]
        #for name,im in self.images.items():
        #    if name != "identity":
        #        self.kernel_stack = np.vstack((self.kernel_stack, im))

    def compile_features(self, selection=None, return_labels=False, images=None, min_number_features=1, number_features=5):
        if images is not None:
            self.images = images

        #if not self.features:
        self.set_default_features(min_number_features=min_number_features, number_features = number_features)

        if not self.lineG:
            G = self.construct_line_graph()
        else:
            G = self.lineG

        arc_map = {}
        arc_pixel_map = {}
        for i, arc in enumerate(self.arcs):
            index = tuple(arc.node_ids) + (len(arc.line),)
            arc_map[index] = i
        # G = self.construct_dual_graph()

        def factors(x):
            factor_tuples = {}
            for i in range(1, 1000):
                if np.abs( int(x/float(i)) - x/float(i) ):
                    factor_tuples[i] = x/float(i)
            return factor_tuples



        # perform neg sampling study
        neg_vec = []
        pos_vec = []

        print("% obtaining arc kernels")

        arc_features = []
        feature_names = []

        semi_sup = False

        for arc in self.arcs:
            index = tuple(arc.node_ids) + (len(arc.line),)
            i = arc_map[index]

            arc_feature_row = []
            
            for image_name, image in self.images.items():

                if i not in arc_pixel_map:
                    arc_pixel_map[i] = get_pixel_values_from_arcs([arc], image)
                arc_pixels = arc_pixel_map[i]

                connected_pixels = []
                for j in G.neighbors(i):
                    arc = self.arcs[j]
                    if j not in arc_pixel_map:
                        arc_pixel_map[j] = get_pixel_values_from_arcs(
                            [arc], self.image
                        )
                    j_pixels = arc_pixel_map[j]
                    connected_pixels.append(j_pixels)

                for function_name, foo in self.features.items():
                    if function_name.startswith("neighbor_"):
                        arc_feature_row.append(foo(connected_pixels))
                    else:
                        arc_feature_row.append(foo(arc_pixels))
                    if len(arc_features) == 0:
                        feature_names.append(image_name + "_" + function_name)

                # if labeled append label for semi-supervised

                if semi_sup:
                    for i in range(1):
                        if arc.label_accuracy is not None and arc.z != 1:

                            arc_feature_row.append(arc.label_accuracy)

                            #if arc.label_accuracy > 0.5:
                            #    arc_feature_row.append(1.0)
                            #if arc.label_accuracy < 0.5:
                            #    arc_feature_row.append(0.0 )
                            #else:
                            #    arc_feature_row.append(.5 )


                            if len(arc_features) == 0:

                                feature_names.append(image_name + "_label_acc_pos"+str(i) )
                                #feature_names.append(image_name + "_PosSig" )
                                #feature_names.append(image_name + "_Label")
                        else:

                            rand_label = np.random.random()
                            arc_feature_row.append(rand_label)
                            #arc_feature_row.append(rand_label)
                            if len(arc_features) == 0:
                                #feature_names.append(image_name + "_NegativeTestLabel")
                                feature_names.append(image_name + "_")

                        if arc.label_accuracy > 0.8:
                            pos_vec = arc_feature_row
                        if arc.label_accuracy < 0.2:
                            neg_vec = arc_feature_row

            n1 , n2 = arc.nodes
            n1_arcs = n1.arcs
            n2_arcs = n2.arcs
            deg = len(n1_arcs)
            num_adj_arc_feats = 2
            for adj_arc in n1_arcs:
                # cosine similarity of adjacent arcs
                cos_sim = cosine_similarity(arc.line, adj_arc.line)
                arc_feature_row.append(cos_sim)
                #deg += 1
                if len(arc_features) == 0:
                    feature_names.append("adjacency_cosine_similarity"+str(cos_sim))

                # gradiant of distance between node most distant arc points in adjacent arcs
                # on a Poincare ball, so the gradient of dtances in hyperbolic space
                adj_n1, adj_n2 = adj_arc.nodes
                adj_node = adj_n2 if adj_n2 != n1 else adj_n1
                hyperbolic_dist = hyperbolic_distance(adj_node.xy, n1.xy)
                #print("ad n1 node xy: ", adj_n1.xy, " other: ", n1.xy)
                #print("ad n2 node xy: ", adj_n2.xy, " other: ", n2.xy)
                #print(hyperbolic_dist)
                arc_feature_row.append(hyperbolic_dist)
                if len(arc_features) == 0:
                    feature_names.append("grad_hyperbolic_dist"+str(hyperbolic_dist))

            # degree of vertex
            arc_feature_row.append(deg)
            if len(arc_features) == 0:
                feature_names.append("degree")

            for i in range( (num_adj_arc_feats * deg) + 1, (num_adj_arc_feats * self.msc.max_degree) + 1 ):
                arc_feature_row.append(0)
                if len(arc_features) == 0:
                    feature_names.append("fill")

            deg = len(n2_arcs)
            for adj_arc in n2_arcs:
                cos_sim = cosine_similarity(arc.line, adj_arc.line)
                arc_feature_row.append(cos_sim)
                #deg += 1
                if len(arc_features) == 0:
                    feature_names.append("adjacency_cosine_similarity"+str(cos_sim))

                adj_n1, adj_n2 = adj_arc.nodes
                adj_node = adj_n1 if adj_n1 != n2 else adj_n2
                hyperbolic_dist = hyperbolic_distance(adj_node.xy, n2.xy)
                #print(hyperbolic_dist)
                arc_feature_row.append(hyperbolic_dist)
                if len(arc_features) == 0:
                    feature_names.append("grad_hyperbolic_dist" + str(hyperbolic_dist))

            # degree of vertex
            arc_feature_row.append(deg)
            if len(arc_features) == 0:
                feature_names.append("degree")

            for i in range( (num_adj_arc_feats * deg) + 1, (num_adj_arc_feats * self.msc.max_degree) + 1 ):
                arc_feature_row.append(0)
                if len(arc_features) == 0:
                    feature_names.append("fill")


            #for node_id in arc.node_ids:
            #    if self.msc.nodes[node_id].index == 1:
            #        saddle_value = self.msc.nodes[node_id].value
            #    else:
            #        maximum_value = self.msc.nodes[node_id].value


            arc_features.append(arc_feature_row)

        arc_features = np.array(arc_features).astype(dtype=np.float32)

        print(" >>>> ")
        print("mu and std of features")
        mu = np.mean(arc_features, axis=0)
        std = np.std(arc_features, axis=0)
        #print(mu)
        #print(std)
        print(" >> ")
        #arc_features = (arc_features - mu) / std   #performs much worse

        if semi_sup:
            print(" >>>> ")
            print("feature vec inner product u^T v ")
            #feature vec inner product u^T v 48670.70808013559
            uv = np.transpose(np.array(pos_vec)).dot(np.array(neg_vec))
            print(uv)
            print("u^T * v + log(3/2)")
            print("-log(k) = ")
            lk = uv + np.log10(3/2)
            print(lk)
            print("Factors: ")
            print(factors(lk))




        self.number_features = len(feature_names)

        if selection is not None:

            selected_arc_indices = []
            for i, arc in enumerate(selection):
                index = tuple(arc.node_ids) + (len(arc.line),)
                selected_arc_indices.append(arc_map[index])
            if return_labels:
                return arc_features[selected_arc_indices, :], feature_names
            else:
                return arc_features[selected_arc_indices, :]

        if return_labels:
            print(">>>> returning with labels")
            return arc_features, feature_names
        else:
            print(" returning without labels")
            self.compiled_features = arc_features
            return arc_features

    def update_training_from_inference(self):
        selector = ArcSelector(self.image, self.msc)

        in_arc_ids, in_arcs, out_arc_ids, out_arcs, out_pixels, selected_test_arcs = selector.launch_ui(msc=self.msc
                                                                                    , use_inference=True)  # xlims=X, ylims=Y)

        self.new_selected_positive_arc_ids = set(in_arc_ids)
        self.new_selected_negative_arc_ids = set(out_arc_ids)
        self.new_selected_positive_arcs = set(in_arcs)  # _ids
        self.new_selected_negative_arcs = set(out_arcs)  # _ids
        self.new_selected_test_arcs = set(selected_test_arcs)

        self.selected_positive_arc_ids = self.new_selected_positive_arc_ids.union(self.selected_positive_arc_ids)
        self.selected_negative_arc_ids = self.new_selected_negative_arc_ids.union(self.selected_negative_arc_ids)
        self.selected_positive_arcs = self.new_selected_positive_arcs.union(self.selected_positive_arcs)
        self.selected_negative_arcs = self.new_selected_negative_arcs.union(self.selected_negative_arcs)
        self.selected_test_arcs = self.new_selected_test_arcs.union(self.selected_test_arcs)

    def select_geomsc_training(self, image=None, X=None, Y=None, fname_selection=None):

        if image is None:
            image = self.image
        #self.raw_image = io.imread(fname, as_gray=True)
        #fname_raw = fname_base + "_smoothed.raw"
        #blurred_image.tofile(fname_raw)
        selector = ArcSelector(image, self.msc)

        if fname_selection is not None:
            in_arcs, out_arcs, out_pixels = selector.write_image(fname_selection)
        else:
            in_arc_ids, in_arcs, out_arc_ids, out_arcs, out_pixels, selected_test_arcs = selector.launch_ui(msc=self.msc)#xlims=X, ylims=Y)

        self.selected_positive_arc_ids = in_arc_ids
        self.selected_negative_arc_ids = out_arc_ids
        self.selected_positive_arcs = in_arcs#_ids
        self.selected_negative_arcs = out_arcs#_ids
        self.selected_test_arcs = selected_test_arcs

        print(" >>>> type from selection : " , type(self.negative_arcs))

        #in_selection, out_selection = self._extract_selection(
        #    in_arcs, out_arcs, out_pixels
        #)
        # self._build_models(in_selection, out_selection)
        return (in_arcs, out_arcs, out_pixels)

    def msc_subgraph_splits(self, validation_samples, validation_hops
                                    , test_samples, test_hops
                                    , X, Y
                                    , accuracy_threshold=0.1, sigmoid=False, multiclass=False
                                    , select_training_from_image=None
                            ,use_infered_graph = False
                            , update_training_from_inference = None
                                    , msc=None,test_graph=False,min_number_features=1,  number_features=5):
        if msc is not None:
            self.assign_msc(msc)

        print(" %% computing image kernels for arc features ")
        # collect/compute features before partition
        #if not self.use_inference:
        if self.compiled_features is None:
            self.compiled_data = self.compile_features(min_number_features=min_number_features
                                                      , number_features=number_features)
            self.compiled_features = self.compiled_data
        else:
            self.compiled_data = self.compiled_features

        if self.use_inference:
            #self.compiled_data = self.features
            #self.selected_positive_arcs = set()
            #self.selected_negative_arcs = set()
            self.update_training_from_inference()

        print("% kernels complete")

        self.positive_arcs = set()
        self.negative_arcs = set()
        self.validation_set = {}
        self.training_set = {}
        self.test_set = {}
        self.validation_set_ids = {}
        self.test_set_ids = {}



        def fill_set(list):
            new_set = set()
            for s in list:
                new_set.add(s)
            return new_set



        if select_training_from_image is not None and not self.use_inference:
            #self.selected_positive_arcs = set()
            #self.selected_negative_arcs = set()
            self.select_geomsc_training(image=select_training_from_image)  # , X=[0,X], Y=[0,Y])

        # could add class for high/mid accuracy arcs

        print(" collecting validation/test subgraph ")

        # use cvt sampling to obtain validation/test edges
        self.sample_graph_neighborhoods(X,Y,count=validation_samples
                                            , rings=validation_hops
                                         , accuracy_threshold=accuracy_threshold, seed=123
                                         , validation=True)
        if test_samples != 0 and test_hops != 0:
            self.sample_graph_neighborhoods(X, Y, count=test_samples
                                            , rings=test_hops
                                            , accuracy_threshold=accuracy_threshold, seed=666
                                            , test=True)
            all_test = self.test_set_ids["positive"].union(self.test_set_ids["negative"])

        print("subgraph split complete")

        #val_and_test = self.validation_set["positive"].union(self.validation_set["negative"]).union(self.test_set["positive"]).union(self.test_set["negative"])
        all_validation = self.validation_set_ids["positive"].union(self.validation_set_ids["negative"])



        #node_map = self.msc.build_node_map()
        #self.sort_labeled_arcs(accuracy_threshold=accuracy_threshold)

        node_map = {}
        if not self.lineG:
            G = self.construct_line_graph()
        else:
            G = self.lineG

        current_arc_id = 0
        node_ids = {}
        node_labels = {}




        i_val = 0  ##!!!! need to set size for val set

        for arc, features in zip(self.arcs, self.compiled_data):
            index = tuple(arc.node_ids) + (len(arc.line),)
            for node_id in arc.node_ids:
                #
                # !! need to fix here to accomadate only ascending ridges
                #
                if node_id not in node_map:
                    node_map[node_id] = []
                node_map[node_id].append(current_arc_id)


                if sigmoid:
                    #n = 0
                    #p = float(arc.label_accuracy)
                    #if index in self.negative_arc_ids:
                    #    n = float(arc.label_accuracy)
                    #    p = 0
                    label = [float(arc.label_accuracy)]#[   n, p ]
                else:
                    # assign label [1,0] if edge corresponds to inaccurate segmentation
                    # assign label [0,1] if msc edge corresponds to accurate segmentation
                    label = [
                        int(index in self.negative_arc_ids),
                        int(index in self.positive_arc_ids),
                    ]
                    if multiclass:
                        if arc.exterior:
                            label = [
                                0,
                                0,
                                1
                            ]
                        else:
                            label = [
                                int(index in self.negative_arc_ids),
                               int(index in self.positive_arc_ids),
                                0
                            ]


                node = G.node[current_arc_id]
                node["index"] = arc.node_ids
                node["size"] = len(arc.line)
                node["features"] = features.tolist()

                # labeled nodes assigned as train, test, or val
                if bool(np.sum(label)) and select_training_from_image is None and not self.use_inference:
                    node["label"] = label  # arc.label_accuracy
                    node["label_accuracy"] = arc.label_accuracy
                    node["prediction"] = None
                    modified = 0
                    if test_graph or arc.z == 1:
                        modified = 1
                        node["train"] = False
                        node["test"] = True
                        node["val"] = False
                        node_ids[current_arc_id] = current_arc_id
                        node_labels[current_arc_id] = label
                        continue
                    if index in all_validation:
                        modified = 1
                        node["train"] = False
                        node["test"] = False
                        node["val"] = True
                    elif (test_samples != 0 and test_hops != 0) and index in all_test:
                        node["test"] = True
                        node["val"] = False
                        node["train"] = False
                    else:  # and  i_val < val_count:
                        modified = 1
                        node["train"] = True
                        node["test"] = False
                        node["val"] = False
                else:
                    node["label"] = label  # arc.label_accuracy
                    node["label_accuracy"] = arc.label_accuracy
                    node["prediction"] = None

                    if self.use_inference and index not in self.selected_test_arcs:
                        if not isinstance(arc.prediction, (int, np.integer)):
                            pred = [1, 0] if arc.prediction[0] >= 0.5 else [0, 1]
                        else:
                            pred = [1, 0] if float(arc.prediction) <= 0.5 else [0 , 1]
                        inf_train_label = pred #node["prediction"]
                        node["label"] = inf_train_label

                    modified = 0

                    if index in all_validation:
                        modified = 1
                        node["train"] = False
                        node["test"] = False
                        node["val"] = True
                    elif index not in all_validation and (index in self.selected_negative_arc_ids or index in self.selected_positive_arc_ids
                                                          and index not in self.selected_test_arcs):

                        label = [
                            int(index in self.selected_negative_arc_ids),
                            int(index in self.selected_positive_arc_ids),
                        ]
                        node["label"] = label
                        node["test"] = False
                        node["val"] = False
                        node["train"] = True
                    else:  # and  i_val < val_count:
                        modified = 1
                        node["train"] = False
                        node["test"] = True
                        node["val"] = False



                """Label all non-selected arcs as test"""
                # if not  bool(np.sum(label)):
                # node["test"] = True

                # G.node[current_arc_id] = node

                # current_arc_id += 1
                node_ids[current_arc_id] = current_arc_id
                node_labels[current_arc_id] = label
            current_arc_id += 1

        for arc_id, arc in list(G.nodes_iter(data=True)):  # G.nodes.items():
            for node_id in arc["index"]:
                for connected_arc_id in node_map[node_id]:
                    G.add_edge(arc_id, connected_arc_id)

        data1 = json_graph.node_link_data(G)
        s1 = json.dumps(data1)  # input graph
        s2 = json.dumps(node_ids)  # dict: nodes to ints
        s3 = json.dumps(node_labels)  # dict: node_id to class

        return (data1, node_ids, node_labels, self.compiled_data)  # (s1, s2, s3, compiled_data)

    def msc_persistence_subgraph_split(self, persistence_values, blur
                                    , test_samples, test_hops
                                    , X, Y
                                    , accuracy_threshold=0.1,  msc=None
                                    ,test_graph=False, number_features=5):
        if msc is not None:
            self.assign_msc(msc)

        print(" %% computing image kernels for arc features ")
        # collect/compute features before partition
        compiled_data = self.compile_features(number_features=number_features)
        print("% kernels complete")

        self.positive_arcs = set()
        self.negative_arcs = set()
        self.validation_set = {}
        self.training_set = {}
        self.test_set = {}
        self.validation_set_ids = {}
        self.test_set_ids = {}

        def fill_set(list):
            new_set = set()
            for s in list:
                new_set.add(s)
            return new_set

        node_map = self.msc.build_node_map()
        self.sort_labeled_arcs(accuracy_threshold=accuracy_threshold)
            # could add class for high/mid accuracy arcs

        print(" collecting validation/test lower persistence MSC subgraph ")

        #val_and_test = self.validation_set["positive"].union(self.validation_set["negative"]).union(self.test_set["positive"]).union(self.test_set["negative"])

        all_test=[]
        node_map = {}
        if not self.lineG:
            G = self.construct_line_graph()
        else:
            G = self.lineG

        current_arc_id = 0
        node_ids = {}
        node_labels = {}

        i_val = 0  ##!!!! need to set size for val set
        train_msc = self.msc_collection[(sorted(persistence_values)[-1], blur)]
        train_msc_arc_ids = set()
        val_msc = self.msc_collection[(sorted(persistence_values)[-2], blur)]
        val_msc_arc_ids = set()
        test_msc = self.msc_collection[(sorted(persistence_values)[0], blur)]
        test_msc_arc_ids = set()


        for arc, features in zip(self.arcs, compiled_data):
            index = tuple(arc.node_ids) + (len(arc.line),)
            for node_id in arc.node_ids:
                #
                # !! need to fix here to accomadate only ascending ridges
                #
                if node_id not in node_map:
                    node_map[node_id] = []
                node_map[node_id].append(current_arc_id)

                # assign label [1,0] if edge corresponds to inaccurate segmentation
                # assign label [0,1] if msc edge corresponds to accurate segmentation
                label = [
                    int(index in self.negative_arc_ids),
                    int(index in self.positive_arc_ids),
                ]

                node = G.node[current_arc_id]
                node["index"] = arc.node_ids
                node["size"] = len(arc.line)
                node["features"] = features.tolist()

                # labeled nodes assigned as train, test, or val
                if bool(np.sum(label)):
                    node["label"] = label  # arc.label_accuracy
                    node["label_accuracy"] = arc.label_accuracy
                    node["prediction"] = None
                    in_val = False
                    in_train = False
                    for val_arc in val_msc.arcs:
                        if self.geomsc.nested_arcs(arc, val_arc) and arc.z != 1:
                            modified = 1
                            node["train"] = False
                            node["test"] = False
                            node["val"] = True
                            arc.partition="val"
                            in_val = True
                        else:
                            modified = 1
                            node["train"] = False
                            node["test"] = True
                            node["val"] = False
                            arc.partition = "test"
                    for train_arc in train_msc.arcs:
                        if self.geomsc.nested_arcs(arc, train_arc) and arc.z != 1:
                            node["test"] = False
                            node["val"] = False
                            node["train"] = True
                            arc.partition="train"
                            in_train = True
                        else:
                            node["train"] = False
                            node["test"] = True
                            node["val"] = False
                            arc.partition = "test"
                    if not in_val and not in_train:
                        node["train"] = False
                        node["test"] = True
                        node["val"] = False
                        arc.partition="test"
                    if arc.z == 1 or test_graph:
                        node["train"] = False
                        node["test"] = True
                        node["val"] = False
                        arc.partition = "test"

                """Label all non-selected arcs as test"""
                # if not  bool(np.sum(label)):
                # node["test"] = True

                # G.node[current_arc_id] = node

                # current_arc_id += 1
                node_ids[current_arc_id] = current_arc_id
                node_labels[current_arc_id] = label
            current_arc_id += 1

        for arc_id, arc in list(G.nodes_iter(data=True)):  # G.nodes.items():
            for node_id in arc["index"]:
                for connected_arc_id in node_map[node_id]:
                    G.add_edge(arc_id, connected_arc_id)

        data1 = json_graph.node_link_data(G)
        s1 = json.dumps(data1)  # input graph
        s2 = json.dumps(node_ids)  # dict: nodes to ints
        s3 = json.dumps(node_labels)  # dict: node_id to class

        return (data1, node_ids, node_labels, compiled_data)
        # (s1, s2, s3, compiled_data)

    def msc_inference_graph(self, persistence_values, blur
                                    , test_samples, test_hops
                                    , X, Y
                                    , accuracy_threshold=0.1,  msc=None
                                    ,test_graph=False):
        if msc is not None:
            self.assign_msc(msc)

        print(" %% computing image kernels for arc features ")
        # collect/compute features before partition
        compiled_data = self.compile_features()
        print("% kernels complete")

        self.positive_arcs = set()
        self.negative_arcs = set()
        self.validation_set = {}
        self.training_set = {}
        self.test_set = {}
        self.validation_set_ids = {}
        self.test_set_ids = {}

        def fill_set(list):
            new_set = set()
            for s in list:
                new_set.add(s)
            return new_set

        node_map = self.msc.build_node_map()
        self.sort_labeled_arcs(accuracy_threshold=accuracy_threshold)
            # could add class for high/mid accuracy arcs

        print(" collecting validation/test lower persistence MSC subgraph ")

        #val_and_test = self.validation_set["positive"].union(self.validation_set["negative"]).union(self.test_set["positive"]).union(self.test_set["negative"])

        all_test=[]
        node_map = {}
        if not self.lineG:
            G = self.construct_line_graph()
        else:
            G = self.lineG

        current_arc_id = 0
        node_ids = {}
        node_labels = {}

        i_val = 0  ##!!!! need to set size for val set
        train_msc = self.msc_collection[(sorted(persistence_values)[-1], blur)]
        train_msc_arc_ids = set()
        val_msc = self.msc_collection[(sorted(persistence_values)[-2], blur)]
        val_msc_arc_ids = set()
        test_msc = self.msc_collection[(sorted(persistence_values)[0], blur)]
        test_msc_arc_ids = set()


        for arc, features in zip(self.arcs, compiled_data):
            index = tuple(arc.node_ids) + (len(arc.line),)
            for node_id in arc.node_ids:
                #
                # !! need to fix here to accomadate only ascending ridges
                #
                if node_id not in node_map:
                    node_map[node_id] = []
                node_map[node_id].append(current_arc_id)

                # assign label [1,0] if edge corresponds to inaccurate segmentation
                # assign label [0,1] if msc edge corresponds to accurate segmentation
                label = [
                    int(index in self.negative_arc_ids),
                    int(index in self.positive_arc_ids),
                ]

                node = G.node[current_arc_id]
                node["index"] = arc.node_ids
                node["size"] = len(arc.line)
                node["features"] = features.tolist()

                # labeled nodes assigned as train, test, or val
                if bool(np.sum(label)):
                    node["label"] = label  # arc.label_accuracy
                    node["label_accuracy"] = arc.label_accuracy
                    node["prediction"] = None
                    in_val = False
                    in_train = False
                    for val_arc in val_msc.arcs:
                        if self.geomsc.nested_arcs(arc, val_arc):
                            modified = 1
                            node["train"] = False
                            node["test"] = False
                            node["val"] = True
                            arc.partition="val"
                            in_val = True
                    for train_arc in train_msc.arcs:
                        if self.geomsc.nested_arcs(arc, train_arc):
                            node["test"] = False
                            node["val"] = False
                            node["train"] = True
                            arc.partition="train"
                            in_train = True
                    if not in_val and not in_train:
                        node["train"] = False
                        node["test"] = True
                        node["val"] = False
                        arc.partition="test"
                    if test_graph:
                        node["train"] = False
                        node["test"] = True
                        node["val"] = False

                """Label all non-selected arcs as test"""
                # if not  bool(np.sum(label)):
                # node["test"] = True

                # G.node[current_arc_id] = node

                # current_arc_id += 1
                node_ids[current_arc_id] = current_arc_id
                node_labels[current_arc_id] = label
            current_arc_id += 1

        for arc_id, arc in list(G.nodes_iter(data=True)):  # G.nodes.items():
            for node_id in arc["index"]:
                for connected_arc_id in node_map[node_id]:
                    G.add_edge(arc_id, connected_arc_id)

        data1 = json_graph.node_link_data(G)
        s1 = json.dumps(data1)  # input graph
        s2 = json.dumps(node_ids)  # dict: nodes to ints
        s3 = json.dumps(node_labels)  # dict: node_id to class

        return (data1, node_ids, node_labels, compiled_data)
        # (s1, s2, s3, compiled_data)

    def draw_segmentation(self, path_name, image, type=None):
        geomsc = self.msc
        geomsc.draw_segmentation(filename=path_name
                                 , X=image.shape[1], Y=image.shape[2]
                                 , reshape_out=False, dpi=164
                                 , valley=True, ridge=True, original_image=image
                                 , type=type)

    def equate_graph(self, G):
        geomsc = self.msc
        self.G = G
        if geomsc is not None:
            geomsc.equate_graph(G)
        self.msc = geomsc
        return self.msc
