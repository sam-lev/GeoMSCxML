# Standard library imports
import sys
import os

# Third party imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import matplotlib.colors as pltcolor
from skimage import io
import imageio
from PIL import Image

import numpy as np
import scipy
from skimage import morphology
import samply

# Local application imports
from topoml.ui.colors import red, blue, green, orange, purple, offIvory
orange = red
from topoml.image.utils import (
    bounding_box,
    box_intersection,
    make_dilated_arc_image,
    make_arc_mask,
    make_mc_arc_mask,
)
from topoml.topology.utils import is_ridge_arc, is_valley_arc


from matplotlib.widgets import LassoSelector, RectangleSelector
from matplotlib.path import Path
from matplotlib.colors import to_rgba


class SelectFromCollection:
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.5):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts, closed=False)
        self.ind = np.nonzero(path.contains_points(self.xys, radius=1.0))[0]
        #self.fc[:, -1] = self.alpha_other
        #self.fc[self.ind, -1] = 1
        #self.collection.set_facecolors(self.fc)
        #self.canvas.draw_idle()

    def disconnect(self, fc):
        #self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(fc)
        self.collection.set_alpha(0.5)
        #self.canvas.draw_idle()




def make_arc_id(a):
    return tuple(a.node_ids) + (len(a.line),)

class ArcSelector(object):
    def __init__(
            self, image=None, msc=None, selection_radius=4, valley=True 
            , ridge=True, invert=False, kdtree=True
    ):
        # Needed for kdTree on large point sets:
        sys.setrecursionlimit(10000)

        self.image = image
        self.msc = msc
        self.msc.arcs = msc.arcs

        self.invert = invert
        
        self.selection_radius = selection_radius
        self.selection_shape = morphology.disk(self.selection_radius)
        self.fat_mask = make_dilated_arc_image(
            self.image, self.msc, self.selection_radius, self.invert
        )

        self.in_color = red
        self.out_color = blue
        self.bg_color = purple

        self.in_arcs = set()
        self.out_arcs = set()
        self.test_arcs = set()
        self.train_arcs = set()

        self.out_pixels = set()
        self.arc_map = []
        self.arc_drawings = {}
        self.arc_points = np.array([0,0])
        self.use_valley_arcs = valley
        self.use_ridge_arcs = ridge

        # This ensures smaller arcs take precedence in the event of
        # pixel overlap
        sorted_arcs = sorted(self.msc.arcs, key=lambda arc: len(arc.line))
        arc_points = []
        self.arcs = []
        for arc in sorted_arcs:
            if (not self.use_ridge_arcs and is_ridge_arc(arc, self.msc)) or (
                not self.use_valley_arcs and is_valley_arc(arc, self.msc)
            ):
                continue
            index = make_arc_id(arc)
            arc_points.extend(arc.line)
            self.arc_map.extend([index] * len(arc.line))
            self.arcs.append(arc)
        # only needed for selection ui to choose neighboring arcs
        # can cause error with sparse MSC
        self.kdtree = None
        if kdtree:
            self.kdtree = scipy.spatial.KDTree(arc_points, leafsize=1000)

    def get_closest_arc_index(self, point):
        distance, index = self.kdtree.query(point)
        return self.arc_map[index]





    def launch_ui(self, msc=None, xlims=None, ylims=None, use_inference = False, box_select = False, msc_arcs = None):
        if msc is not None:
            self.msc = msc

        if xlims is None or ylims is None and self.image is not None:
            X = self.image.shape[0]
            Y = self.image.shape[1]
            xlims = [0, X]
            ylims = [0, Y]

        plt.ion()

        if xlims is None or ylims is None:
            subplot_kw = dict(xlim=(0, self.image.shape[1]), ylim=(self.image.shape[0], 0), autoscale_on=False)
        if xlims is not None and ylims is not None:
            subplot_kw = dict(xlim=(xlims[0], xlims[1]), ylim=(ylims[1], ylims[0]), autoscale_on=False)
        self.fig, self.ax= plt.subplots(subplot_kw=subplot_kw) #figure()

        arc_xpoints , arc_ypoints = [] , []
        arc_points = []
        self.scatter_points = self.ax.scatter(arc_xpoints,
                                            arc_ypoints,
                                            facecolor="ivory",
                                            edgecolor="none",
                                            s=1,
                                            marker=",",
                                            alpha=0.3,
                                            zorder=1,
            )


        #self.ax.set_xlim(0, self.image.shape[1]) #plt.gca().set_xlim(
        #self.ax.set_ylim(self.image.shape[0], 0)
        #if xlims is not None:
        #    self.ax.set_xlim(xlims[0], xlims[1])
        #if ylims is not None:
        #    self.ax.set_ylim(ylims[1], ylims[0])

        if use_inference:
            cmap = cm.get_cmap('seismic')
            cmap.set_under('black')
            cmap.set_bad('black')
            # cmap.set_over('white')
            #plt.set_cmap(cmap)

            cmap_accurate = cm.get_cmap('cool')

        plt.imshow(self.image, cmap=plt.cm.Greys_r, zorder=2) # cmap=plt.cm.Greys_r, #cmap=plt.cm.Greys_r,
        c = 0
        for arc in self.msc.arcs:
            if (not self.use_ridge_arcs and is_ridge_arc(arc, self.msc)) or (
                not self.use_valley_arcs and is_valley_arc(arc, self.msc)
            ):
                continue

            arc_index = make_arc_id(arc)
            points = np.array(arc.line)

            arc_xpoints.append(points[:,0])
            arc_ypoints.append(points[:,1])
            for point in points:
                arc_points.append(np.asarray(point))

            old_offset = self.scatter_points.get_offsets()
            new_offset = np.concatenate([old_offset, np.array(points)])
            old_color = self.scatter_points.get_facecolors()

            if c == 0:
                new_color = np.concatenate([old_color, np.array(old_color)])
                c+=1
            else:
                new_color = np.concatenate([old_color, np.array([old_color[0,:]])])

            self.scatter_points.set_offsets(new_offset)#np.c_[points[:,0],points[:,1]])
            self.scatter_points.set_facecolors(new_color)
            #self.fig.canvas.draw()

            color = "ivory"

            if use_inference:
                if not isinstance(arc.prediction, (int, np.integer)):
                    if len(arc.prediction) == 3:
                        label_color = cmap(0.5) if float(arc.prediction[2]) > 0.5 else cmap(float(arc.prediction[1]))
                        pred = float(arc.prediction[2]) if float(arc.prediction[2]) > 0.5 else float(arc.prediction[1])
                    else:
                        label_color = cmap(float(arc.prediction[1]))
                        pred = float(arc.prediction[1])
                else:
                    # print("pred ", arc.prediction)
                    label_color = cmap(float(arc.prediction))
                    pred = float(arc.prediction)
                color = label_color
                if pred <= 0.06 or pred >= 0.94:
                    color = cmap_accurate(pred)

            self.arc_drawings[arc_index] = self.ax.scatter(           ##### plt.scatter
                points[:, 0],
                points[:, 1],
                facecolor=color,
                edgecolor="none",
                s=2,
                marker=",",
                alpha=0.3,
                zorder=3,
            )
        #np.c_[points[:, 0], points[:, 1]])
        self.scatter_points.set_visible(False)
        self.fig.canvas.draw()

        if self.use_ridge_arcs:
            arc_mask = make_mc_arc_mask(self.image, self.msc, False)
            plt.imshow(
                arc_mask,
                cmap="Oranges",
                vmin=0,
                vmax=4,
                interpolation="none",
                alpha=0.3,
                zorder=2,
            )
        if self.use_valley_arcs:
            arc_mask = make_mc_arc_mask(self.image, self.msc, True)
            plt.imshow(
                arc_mask,
                cmap="Blues",
                vmin=0,
                vmax=4,
                interpolation="none",
                alpha=0.3,
                zorder=2,
            )

        print(">>>> image shape: ", self.image.shape)
        extrema_points = [[]] # append all 2-saddle to inner array for slicing
        for node in self.msc.nodes.values():
            x, y = node.xy
            extrema_points[0].append([x, y])

        #for i in extrema_points:#.keys():# , color  enumerate([blue, green, red]):
        color = green
        xy = np.array(extrema_points) #[i])
        self.ax.scatter(                           #plt.scatter
            xy[:, 0],
            xy[:, 1],
            facecolor=color,
            edgecolor="none",
            s=4,
            marker=",",
            zorder=4,
        )

        # add inference results of certain accuracy to training set
        #if use_inference:
        #    self.color_by_predictions()

        #if box_select:
        #    # drawtype is 'box' or 'line' or 'none'
        #    if msc_arcs is not None:
        #        self.msc_arcs = msc_arcs
        #   self.toggle_selector.RS = RectangleSelector(self.ax, self.line_select_callback,
        #                                           drawtype='box', useblit=True,
        #                                           button=[1, 3],  # disable middle button
        #                                           minspanx=5, minspany=5,
        #                                           spancoords='pixels',
        #                                           interactive=True)
        #    self.fig.canvas.mpl_connect('key_press_event', self.toggle_selector)
        #else:
        self.selector = SelectFromCollection(self.ax, self.scatter_points)
        self.fig.canvas.mpl_connect("key_press_event", self.assign_class)
        #self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        plt.show(block=True)

        in_arcs = []
        out_arcs = []
        test_arcs = []
        train_arcs = []
        test_arc_ids = []
        for arc in self.msc.arcs:
            index = make_arc_id(arc)
            if index in self.in_arcs:
                in_arcs.append(arc)
            elif index in self.out_arcs:
                out_arcs.append(arc)
            #elif index in self.train_arcs:
            #    train_arcs.append(arc)
            else:
                test_arcs.append(arc)
                test_arc_ids.append(index)

        #end of ui
        return self.in_arcs, in_arcs, self.out_arcs, out_arcs, np.array(list(self.out_pixels)), test_arcs, test_arc_ids



    def write_image(self, filename):
        self.fig = plt.figure()
        plt.imshow(self.image, cmap=plt.cm.Greys_r, zorder=1)

        for arc in self.msc.arcs:
            if (not self.use_ridge_arcs and is_ridge_arc(arc, self.msc)) or (
                not self.use_valley_arcs and is_valley_arc(arc, self.msc)
            ):
                continue

            arc_index = make_arc_id(arc)
            if arc_index in self.in_arcs:
                points = np.array(arc.line)
                self.arc_drawings[arc_index] = plt.scatter(
                    points[:, 0],
                    points[:, 1],
                    facecolor= self.in_color,
                    edgecolor="none",
                    s=2,
                    marker=",",
                    zorder=3,
                )
                self.arc_drawings[arc_index].set_visible(True)
            elif arc_index in self.out_arcs:
                points = np.array(arc.line)
                self.arc_drawings[arc_index] = plt.scatter(
                    points[:, 0],
                    points[:, 1],
                    facecolor=self.out_color,
                    edgecolor="none",
                    s=2,
                    marker=",",
                    zorder=3,
                )
                self.arc_drawings[arc_index].set_visible(True)

        if self.use_ridge_arcs:
            arc_mask = make_mc_arc_mask(self.image, self.msc, False)
            plt.imshow(
                arc_mask,
                cmap="Oranges",
                vmin=0,
                vmax=4,
                interpolation="none",
                alpha=0.8,
                zorder=2,
            )
        if self.use_valley_arcs:
            arc_mask = make_mc_arc_mask(self.image, self.msc, True)
            plt.imshow(
                arc_mask,
                cmap="Blues",
                vmin=0,
                vmax=4,
                interpolation="none",
                alpha=0.8,
                zorder=2,
            )

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

        plt.gca().set_xlim(0, self.image.shape[1])
        plt.gca().set_ylim(self.image.shape[0], 0)
        plt.savefig(filename)

        in_arcs = []
        out_arcs = []
        for arc in self.msc.arcs:
            index = make_arc_id(arc)
            if index in self.in_arcs:
                in_arcs.append(arc)
            elif index in self.out_arcs:
                out_arcs.append(arc)

        return (self.in_arcs, out_arcs, np.array(list(self.out_pixels)))
    
    def make_to_scale_image(image_in, outputname, size=(1, 1), dpi=80):
        fig = plt.figure()
        fig.set_size_inches(size)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.set_cmap('binary')
        ax.imshow(image_in, aspect='equal')
        plt.savefig(outputname, dpi=dpi)

    def draw_binary_segmentation(self, filename, msc = None, invert=False, reshape_out = True, dpi = True):
        if not msc:
            self.msc = msc
        
        black_box = np.zeros((self.image.shape[0],self.image.shape[1])) if not invert else np.zeros((self.image.shape[1],self.image.shape[0]))
        #print(self.image.shape+(,,3))
        fig = plt.imshow(black_box,cmap='gray', alpha=None,interpolation = 'nearest') #plt.figure() #in
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        for arc in self.msc.arcs:
            if (not self.use_ridge_arcs and is_ridge_arc(arc, self.msc)) or (
                not self.use_valley_arcs and is_valley_arc(arc, self.msc)
            ):
                continue

            arc_index = make_arc_id(arc)
            if arc_index in self.in_arcs:
                points = np.array(arc.line)
                self.arc_drawings[arc_index] = plt.scatter(
                    points[:, 0],
                    points[:, 1],
                    facecolor= 'white',#self.in_color,
                    alpha=0.5,
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
                    facecolor= 'white',#self.out_color,
                    edgecolor="none",
                    alpha=0.5,
                    s=5,
                    marker=",",
                    zorder=3,
                )
                self.arc_drawings[arc_index].set_visible(True)

        if self.use_ridge_arcs:
            arc_mask = make_mc_arc_mask(self.image, self.msc, invert)#False)
            plt.imshow(
                arc_mask,
                cmap="binary",
                vmin=0,
                vmax=0,
                norm=plt.Normalize(vmin=0, vmax=1),
                interpolation="nearest",
                alpha=0.5,
                zorder=2,
            )
        if self.use_valley_arcs:
            arc_mask = make_mc_arc_mask(self.image, self.msc, invert)#True)
            plt.imshow(
                arc_mask,
                cmap="binary",
                vmin=0,
                vmax=0,
                interpolation="nearest",
                alpha=0.5,
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

        #plt.gca().set_axis_off()
        #plt.gca().set_xlim(0, self.image.shape[0])
        #plt.gca().set_ylim(self.image.shape[1], 0)
        dpi_ = None
        if dpi:
            if self.image.shape[0] >= 600:
                dpi_ = 600
            else:
                dpi_ = 156
            if isinstance(dpi, int):
                dpi_ = dpi
        
        plt.savefig(filename, bbox_inches='tight', pad_inches = 0.0, dpi = dpi_, transparent = False, cmap=None)

        img = imageio.imread(filename)
        
        if reshape_out:
            if self.image.shape[0] >= 600:
                img = Image.fromarray(img).resize((img.shape[1]+9, img.shape[1]-91+5))
            else:
                img = Image.fromarray(img).resize((img.shape[1]+1, img.shape[0]+1))
        else:
            img = Image.fromarray(img).resize((img.shape[1]+1, img.shape[0]+1)) #if not invert else Image.fromarray(img).resize((img.shape[0], img.shape[1]))
        img = np.array(img)[:,:,0]
        Img = Image.fromarray(img)
        Img.save(filename)

        plt.close()
        
        in_arcs = []
        out_arcs = []
        for arc in self.msc.arcs:
            index = make_arc_id(arc)
            if index in self.in_arcs:
                in_arcs.append(arc)
            elif index in self.out_arcs:
                out_arcs.append(arc)

        return (in_arcs, out_arcs, np.array(list(self.out_pixels)))


    def draw_segmentation(self, filename, msc=None, valley_vs_ridge=False
                          , invert=False, reshape_out=True, dpi=True, original_image=None):
        if msc is not None:
            self.msc = msc

        black_box = np.zeros((self.image.shape[0], self.image.shape[1])) if not invert else np.zeros(
            (self.image.shape[1], self.image.shape[0]))

        print("image shape in arc select draw segmentation ", self.image.shape)
        # print(self.image.shape+(,,3))
        #cmap = cm.get_cmap('Spectral')
        cmap = cm.get_cmap('RdYlGn')
        cmap.set_under('black')
        cmap.set_bad('black')
        #cmap.set_over('white')
        plt.set_cmap(cmap)
        fig = plt.imshow(black_box, cmap=cmap, alpha=None, vmin=0)#, interpolation='nearest')  # plt.figure() #in
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)


        for arc in self.msc.arcs:

            label_color = cmap(arc.label_accuracy)

            if valley_vs_ridge and ((not self.use_ridge_arcs and is_ridge_arc(arc, self.msc)) or (
                    not self.use_valley_arcs and is_valley_arc(arc, self.msc))
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
                self.arc_drawings[arc_index].set_visible(True)
        if valley_vs_ridge == False:
            self.use_ridge_arcs = True
            self.use_valley_arcs = True
        if self.use_ridge_arcs:
            arc_mask = make_mc_arc_mask(self.image, self.msc, invert)  # False)
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
            arc_mask = make_mc_arc_mask(self.image, self.msc, invert)  # True)
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
            if self.image.shape[0] >= 600:
                dpi_ = 600
            else:
                dpi_ = 156
            if isinstance(dpi, int):
                dpi_ = dpi

        plt.savefig(filename, bbox_inches='tight', pad_inches=0.0, dpi=dpi_, transparent=False, cmap=cmap)

        img = imageio.imread(filename)

        if False:#reshape_out:
            if self.image.shape[0] >= 600:
                img = Image.fromarray(img).resize((img.shape[1] + 9, img.shape[1] - 91 + 5))
            else:
                img = Image.fromarray(img).resize((img.shape[1] + 1, img.shape[0] + 1))
        else:
            img = img#Image.fromarray(img).resize((img.shape[1] + 1, img.shape[
                #0] + 1))  # if not invert else Image.fromarray(img).resize((img.shape[0], img.shape[1]))
        if original_image is not None:
            x =np.array(img).shape[0]
            y = np.array(img).shape[1]
            mapped_img = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3))#,np.float32)
            mapped_img[:,:,0] = np.array(img)[:x,:y,0]/255.+np.array(original_image)[:x,:y,0]/255.#[:, :, 0]
            mapped_img[:, :, 1] = np.array(img)[:x, :y, 1]/255. + np.array(original_image)[:x, :y, 1]/255.  # [:, :, 0]
            mapped_img[:, :, 2] = np.array(img)[:x, :y, 2]/255. + np.array(original_image)[:x, :y, 2]/255.  # [:, :, 0]
            print("im shape msc ", np.array(img).shape, " pg img ", np.array(original_image).shape)
            print("mapped im shape ", mapped_img.shape)
            mapped_img*= 255
            Img = Image.fromarray(mapped_img.astype('uint8'))#mapped_img)
            Img.save(filename+'MAP.'+filename.split('.')[-1])

        #img = np.array(img)[:, :, 0]
        #Img = Image.fromarray(img)
        #Img.save(filename) #saves as grey scale

        plt.close()

        in_arcs = []
        out_arcs = []
        test_arcs = []
        for arc in self.msc.arcs:
            index = make_arc_id(arc)
            if index in self.in_arcs:
                in_arcs.append(arc)
            elif index in self.out_arcs:
                out_arcs.append(arc)
            elif index in self.test_arcs:
                test_arcs.append(arc)

        return (in_arcs, out_arcs, np.array(list(self.out_pixels))), test_arcs

    def toggle_arc_lasso(self, x_points, y_points, event_key='1'):
        in_class = event_key == '1'
        out_class = event_key == '2'
        remove_arcs = event_key == 'x'

        min_indices = []
        for x, y in zip(x_points,y_points): # no loop just x y event
            pt = np.array([x, y])
            min_index = self.get_closest_arc_index(pt)
            min_indices.append(min_index)
            if in_class:
                if min_index in self.in_arcs:
                    self.in_arcs.remove(min_index)
                elif min_index in self.out_arcs:
                    self.out_arcs.remove(min_index)
                elif min_index in self.test_arcs:
                    self.test_arcs.remove(min_index)
                #else:
                #    #print("added to positive labels")
                self.in_arcs.add(min_index)
            if out_class:
                if min_index in self.out_arcs:
                    self.out_arcs.remove(min_index)
                elif min_index in self.in_arcs:
                    self.in_arcs.remove(min_index)
                elif min_index in self.test_arcs:
                    self.test_arcs.remove(min_index)
                #    #print("added to negative labels")
                self.out_arcs.add(min_index)
            if remove_arcs:
                if min_index in self.out_arcs:
                    #print("removed from negative labels")
                    self.out_arcs.remove(min_index)
                elif min_index in self.in_arcs:
                    #print("removed from positive labels")
                    self.in_arcs.remove(min_index)
                elif min_index in self.test_arcs:
                    self.test_arcs.remove(min_index)
                self.test_arcs.add(min_index)
        return min_indices                            # return single min_index

    def toggle_arc_box(self, x_points, y_points, ground_truth = None):

        min_indices = []
        for x, y in zip(x_points,y_points): # no loop just x y event
            pt = np.array([x, y])
            min_index = self.get_closest_arc_index(pt)
            min_indices.append(min_index)
            arc = self.msc_arcs[min_index]
            gt_label = arc.ground_truth
            if gt_label == 1:
                if min_index in self.in_arcs:
                    self.in_arcs.remove(min_index)
                elif min_index in self.out_arcs:
                    self.out_arcs.remove(min_index)
                elif min_index in self.test_arcs:
                    self.test_arcs.remove(min_index)
                #else:
                #    #print("added to positive labels")
                self.in_arcs.add(min_index)

                self.arc_drawings[min_index].set_facecolor(self.in_color)
                self.arc_drawings[min_index].set_alpha(0.3)
            if gt_label == 0:
                if min_index in self.out_arcs:
                    self.out_arcs.remove(min_index)
                elif min_index in self.in_arcs:
                    self.in_arcs.remove(min_index)
                elif min_index in self.test_arcs:
                    self.test_arcs.remove(min_index)
                #    #print("added to negative labels")
                self.out_arcs.add(min_index)

                self.arc_drawings[min_index].set_facecolor(self.out_color)
                self.arc_drawings[min_index].set_alpha(0.3)

            self.scatter_points.set_visible(False)  # self.arc_drawings[selected_index].
            #        not self.scatter_points(points[0],points[1]).get_visible()    # self.arc_drawings[selected_index].get_v
            #    )
            # if event.key == "1":
            #    self.selector.disconnect(self.in_color)#self.scatter_points.set_facecolor(self.in_color)
            # elif event.key == "2":
            #    self.selector.disconnect(self.out_color)#self.scatter_points.set_facecolor(self.out_color)
            # else:
            #    self.highlight_pixels(int(event.xdata), int(event.ydata))

            # if event.key == "1":
            #    print(selector.xys[selector.ind])
            #    selector.disconnect()
            #    ax.set_title("")
            self.fig.canvas.draw()
        return min_indices

    def line_select_callback(self, eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        x_set = range(np.min(x1, x2), np.max(x1,x2))
        y_set = range(np.min(y1,y2) , np.max(y1,y2))
        self.toggle_arc_box(x_set, y_set)
        print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        print(f" The buttons you used were: {eclick.button} {erelease.button}")

    def toggle_selector(self, event):
        print(' Key pressed.')
        if event.key == 't':
            if self.toggle_selector.RS.active:
                print(' RectangleSelector deactivated.')
                self.toggle_selector.RS.set_active(False)
            else:
                print(' RectangleSelector activated.')
                self.toggle_selector.RS.set_active(True)

    def toggle_arc_click(self, x, y, key = 1):
        in_class = key == 1
        min_indices = []
        #for x, y in zip(x_points,y_points): # no loop just x y event
        pt = np.array([x, y])
        min_index = self.get_closest_arc_index(pt)
        #min_indices.append(min_index)
        if in_class:
            if min_index in self.in_arcs:
                #print("removed from positive labels")
                self.in_arcs.remove(min_index)
            elif min_index in self.out_arcs:
                #print("removed from negative labels")
                self.out_arcs.remove(min_index)
            else:
                #print("added to positive labels")
                self.in_arcs.add(min_index)
        elif key == 2:
            if min_index in self.out_arcs:
                #print("removed from negative labels")
                self.out_arcs.remove(min_index)
            elif min_index in self.in_arcs:
                #print("removed from positive labels")
                self.in_arcs.remove(min_index)
            else:
                #print("added to negative labels")
                self.out_arcs.add(min_index)
        else:
            if min_index in self.out_arcs:
                #print("removed from negative labels")
                self.out_arcs.remove(min_index)
            elif min_index in self.in_arcs:
                #print("removed from positive labels")
                self.in_arcs.remove(min_index)
        return min_indices                            # return single min_index

    def highlight_pixels(self, x, y):
        start_y = y - self.selection_radius
        start_x = x - self.selection_radius
        for i in range(0, 2 * self.selection_radius + 1):
            xi = start_x + i
            if xi > 0 and xi < self.fat_mask.shape[1]:
                for j in range(0, 2 * self.selection_radius + 1):
                    yj = start_y + j
                    if yj > 0 and yj < self.fat_mask.shape[0]:
                        if (
                            self.selection_shape[j, i]
                            and not self.fat_mask[yj, xi]
                        ):
                            self.out_pixels.add((xi, yj))

    def on_click(self, event):
        if event is None or None in [event.xdata, event.ydata, event.button]:
            return

        if self.fat_mask[int(event.ydata), int(event.xdata)]:
            selected_index = self.toggle_arc_click(
                event.xdata, event.ydata, event.button == 1
            )
            self.arc_drawings[selected_index].set_visible(
                not self.arc_drawings[selected_index].get_visible()
            )
            if event.button == 1:
                self.arc_drawings[selected_index].set_facecolor(self.in_color)
            else:
                self.arc_drawings[selected_index].set_facecolor(self.out_color)
        else:
            self.highlight_pixels(int(event.xdata), int(event.ydata))

        if len(self.out_pixels):
            bg_points = np.array(list(self.out_pixels))
            plt.scatter(
                bg_points[:, 0],
                bg_points[:, 1],
                edgecolor="none",
                facecolor=self.bg_color,
                s=1,
                marker=",",
                zorder=4,
            )
        self.fig.canvas.draw()

    def assign_class(self, event):
        xdata = self.selector.xys[self.selector.ind][:,0]
        ydata = self.selector.xys[self.selector.ind][:, 1]

        #print(" >>>> points: ")
        #print(self.selector.xys[self.selector.ind])
        #print(" x : ", self.selector.xys[self.selector.ind][:,0])
        #if self.fat_mask[int(ydata), int(xdata)]:
        selected_indices = self.toggle_arc_lasso(                         #           selected_index
            self.selector.xys[self.selector.ind][:,0],self.selector.xys[self.selector.ind][:,1]
            , event.key  #event.xdata
        )
        for selected_index in selected_indices:
            #self.arc_drawings[selected_index].set_visible(
            #    not self.arc_drawings[selected_index].get_visible()
            #)
            if event.key == '1':
                self.arc_drawings[selected_index].set_facecolor(self.in_color)
                self.arc_drawings[selected_index].set_alpha(0.3)
            elif event.key == '2':
                self.arc_drawings[selected_index].set_facecolor(self.out_color)
                self.arc_drawings[selected_index].set_alpha(0.3)
            elif event.key == 'x':
                self.arc_drawings[selected_index].set_facecolor(offIvory)
                self.arc_drawings[selected_index].set_alpha(0.3)

        self.scatter_points.set_visible( False )                         # self.arc_drawings[selected_index].
        #        not self.scatter_points(points[0],points[1]).get_visible()    # self.arc_drawings[selected_index].get_v
        #    )
        #if event.key == "1":
        #    self.selector.disconnect(self.in_color)#self.scatter_points.set_facecolor(self.in_color)
        #elif event.key == "2":
        #    self.selector.disconnect(self.out_color)#self.scatter_points.set_facecolor(self.out_color)
        #else:
        #    self.highlight_pixels(int(event.xdata), int(event.ydata))

        #if event.key == "1":
        #    print(selector.xys[selector.ind])
        #    selector.disconnect()
        #    ax.set_title("")
        self.fig.canvas.draw()

    def color_by_predictions(self):

        #####cmap = cm.get_cmap('bwr')
        ####cmap.set_under('black')
        ####cmap.set_bad('black')

        # cmap.set_over('white')

        ###plt.set_cmap(cmap)

        #fig = plt.imshow(black_box, cmap=cmap, alpha=None, vmin=0)

        for arc in self.arcs:


            if not isinstance(arc.prediction, (int, np.integer)):
                if len(arc.prediction) == 3:
                    ####label_color = cmap(0.5) if float(arc.prediction[2]) > 0.5 else cmap(float(arc.prediction[1]))
                    pred = float(arc.prediction[2])  if float(arc.prediction[2]) > 0.5 else float(arc.prediction[1])
                else:
                    ####label_color = cmap(float(arc.prediction[1]))
                    pred = float(arc.prediction[1])
            else:
                # print("pred ", arc.prediction)
                ####label_color = cmap(float(arc.prediction))
                pred = float(arc.prediction)

            #self.arc_drawings[selected_index].set_visible(
            #    not self.arc_drawings[selected_index].get_visible()
            #)

            pt = np.array(arc.line[0])
            min_index = self.get_closest_arc_index(pt)
            if pred >= 0.93:
                if min_index in self.in_arcs:
                    self.in_arcs.remove(min_index)
                elif min_index in self.out_arcs:
                    self.out_arcs.remove(min_index)
                self.in_arcs.add(min_index)
            elif pred <= 0.07:
                if min_index in self.out_arcs:
                    self.out_arcs.remove(min_index)
                elif min_index in self.in_arcs:
                    self.in_arcs.remove(min_index)
                self.out_arcs.add(min_index)
            else:
                if min_index in self.test_arcs:
                    self.test_arcs.remove(min_index)
                self.test_arcs.add(min_index)

            ####self.arc_drawings[min_index].set_facecolor(label_color)

        ####self.scatter_points.set_visible( False )                         # self.arc_drawings[selected_index].

       ####self.fig.canvas.draw()

    def save_arcs(self, filename="arcs.csv", mode="a"):
        f = open(filename, mode)
        for index in self.in_arcs:
            f.write("{},{},{},{}\n".format(1, *index))
        for index in self.out_arcs:
            f.write("{},{},{},{}\n".format(0, *index))
        f.close()

    def load_arcs(self, filename="arcs.csv"):
        if os.path.exists(filename):
            f = open(filename, "r")
            for line in f:
                tokens = list(map(int, line.strip().split(",")))
                if tokens[0]:
                    self.in_arcs.add(tuple(tokens[1:]))
                else:
                    self.out_arcs.add(tuple(tokens[1:]))
        return (self.in_arcs, self.out_arcs)

    def cull_selection(self, window):
        for a in self.msc.arcs:
            arc_index = make_arc_id(a)
            for arc_set in [self.in_arcs, self.out_arcs]:
                if arc_index in arc_set:
                    if not box_intersection(window, bounding_box(a.line)):
                        arc_set.remove(arc_index)
        return self.in_arcs, self.out_arcs

    def construct_node_map(self):
        node_map = {}
        current_arc_index = 0
        for arc in self.arcs:
            for node_id in arc.node_ids:
                if node_id not in node_map:
                    node_map[node_id] = []
                node_map[node_id].append(current_arc_index)
            current_arc_index += 1
        return node_map

    def sample_selection(self, count=20, rings=1, seed=0):
        np.random.seed(seed)
        X = samply.hypercube.cvt(count, 2)
        X[:, 0] *= self.image.shape[1]
        X[:, 1] *= self.image.shape[0]

        node_map = self.construct_node_map()
        
        seed_arc_keys = list()

        for x in X:
            arc_key = self.get_closest_arc_index(x)
            seed_arc_keys.append(arc_key)
        ring = 0
        ring_index = 0
        ring_count = len(seed_arc_keys)

        while ring <= rings:
            next_ring = seed_arc_keys[ring_index:(ring_index+ring_count)]

            ring_count = 0
            for arc_key in next_ring:
                for node_id in arc_key[:2]:
                    for arc_index in node_map[node_id]:
                        neighbor_arc_key = make_arc_id(self.arcs[arc_index])
                        if neighbor_arc_key not in seed_arc_keys:
                            seed_arc_keys.append(neighbor_arc_key)
                            ring_count += 1
                ring_index += 1
            ring += 1

        seed_arc_keys = set(seed_arc_keys)
        
        """
        for arc_set in [self.in_arcs, self.out_arcs]:
            for arc_key in self.arc_map:
                if arc_key in arc_set and arc_key not in seed_arc_keys:
                    arc_set.remove(arc_key)
        """
        return (self.in_arcs.intersection(seed_arc_keys), self.out_arcs.intersection(seed_arc_keys))
