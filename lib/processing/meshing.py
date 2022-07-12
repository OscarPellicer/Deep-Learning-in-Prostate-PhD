#Base python libraries
import sys, os, copy, glob
from joblib import Parallel, delayed
from functools import partial
from collections import defaultdict

#Base scientific libraries
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns 
from scipy.interpolate import RBFInterpolator

#Other imaging / mesh-related libraries
from pyvista.core.pointset import UnstructuredGrid
import SimpleITK as sitk
import pydicom
import vtk

#Important functions
def get_FEM_displacements(mesh, surface_node_idx, surface_displacements, time_steps=4, 
                          material='neo-Hookean', **material_properties):
    '''
        Computes the displacement field within a mesh given some known displacements
    '''
    from ..febio import FEBioModel

    #Create model
    log_data= ['ux', 'uy', 'uz']
    m= FEBioModel(log_data=log_data)
    m.add_step(name='displacement', time_steps=str(time_steps), step_size=str(1/time_steps), 
               time_stepper={'dtmin': '0.01', 'dtmax': str(1/time_steps), 'max_retries': '5', 'opt_iter': '10'})
    m.add_load_curve(name='displacement_lc')
    m.add_object(mesh.points, mesh.cells_dict['tetra'], 'tet4', name='prostate')

    #Add boundary conditions
    for i, axis in enumerate(['x', 'y', 'z']):
        m.add_node_scalar_field(surface_node_idx, surface_displacements[:,i], 
                                object_name='prostate', name=f'disp_u{axis}', node_set_name=f'surface_nodes')
        m.add_displacement('displacement', axis, 'map', f'disp_u{axis}', 'displacement_lc')

    #Add material, run simulation, and read results
    m.add_material(material, 'prostate', **material_properties)
    results= m.run(steps=[time_steps]) #Get only the results from the last step
    fem_displacements= np.stack([results[time_steps][u] for u in log_data], axis=-1)
    
    return fem_displacements

def rigid_transform_from_points(P, Q, translation_only=False):
    '''
        For matrix P = [p1 p2] and Q = [q1 q2], Find optimal M such that Q = RP and R is an orthogonal matrix
        Source: http://nghiaho.com/?page_id=671
        P and Q are matrices of N points with shape Nx3
    '''
    #Compute optimal transform
    P, Q= np.array(P).T, np.array(Q).T
    assert P.shape == Q.shape
    Pc, Qc= np.mean(P, axis=1, keepdims=True), np.mean(Q, axis=1, keepdims=True)
    H= (P-Pc) @ (Q-Qc).T
    U, S, V= np.linalg.svd(H)
    R= V.T @ U.T
    if np.linalg.det(R) < 0:
        V[2,:]*= -1
        R = V.T @ U.T
    t= -R @ Pc + Qc
    P_after= R @ P + t
    
    #To sitk
    tfm_sitk = sitk.Similarity3DTransform()
    if not translation_only:
        tfm_sitk.SetMatrix(R.flatten())
        tfm_sitk.SetTranslation(t[:,0])
    else:
        tfm_sitk.SetMatrix(np.eye(3).flatten())
        tfm_sitk.SetTranslation((Qc-Pc)[:,0])
    tfm_sitk.SetCenter([0]*3)
    Ps= [tfm_sitk.TransformPoint(p) for p in P.T]
    
    return tfm_sitk

def get_DDF(points, deformations, size, spacing, n_jobs=-1, **rbf_kwargs):
    '''
        Predict a 3D dense deformation field (DDF) by using a sample of known deformations
        in that grid and interpolating the rest with an RBF interpolator
        
        Parameters
        ----------
        points: array
            N x 3 array of positions where the deformation field is known
        deformations: array
            N X D array of deformations associated with points in surface_points
        size: list or array
            Size of the output DDF. It must be coherent with the image from which the 
            points and deformations were obtained
        spacing: list or array
            Spacing of the output DDF. It must be coherent with the image from which the 
            points and deformations were obtained
        n_jobs: int, default -1
            Number of threads for predicting on the DDF grid, default -1: Use all threads
        **rbf_kwargs: **dict
            Parameters for the RBFInterpolator
            
        Returns
        -------
        DDF_sitk_tfm: SimpleITK Transform
    '''
    #Pre-process
    size, spacing= np.array(size), np.array(spacing)
    
    #Grid of point idx over which to prdict DDF (spacing is NOT considered here)
    xyz= [np.arange(0, size[i]) for i in range(len(size))]
    xyz_g = np.meshgrid(*xyz, indexing='xy')

    #Predict on the center of the voxel (shifted 0.5) and consider spacing
    all_points= (np.stack(xyz_g).reshape(3,-1).T + 0.5) * spacing

    #Predict on the grid
    DDF_fn= RBFInterpolator(points, deformations, **rbf_kwargs)
    if n_jobs in [0,1]:
        DDF_out= DDF_fn(all_points) 
    else:
        n_jobs= os.cpu_count() if n_jobs < 0 else min(os.cpu_count(), n_jobs)
        DDF_out= np.concatenate( Parallel(n_jobs=n_jobs)(
            delayed(DDF_fn)(arr) for arr in np.array_split(all_points, n_jobs) ) )
    DDF_shape= (*size[::-1], len(size))
    
    #Reshape
    DDF_flat= copy.deepcopy(DDF_out) #/ mr_img.GetSpacing()[0]
    DDF= DDF_flat.reshape(DDF_shape).swapaxes(0,1).swapaxes(0,2)
    
    #Get the sitk DDF transform too
    DDF_sitk= sitk.GetImageFromArray(DDF, isVector=True)
    DDF_sitk.SetSpacing(spacing)
    DDF_sitk= sitk.Cast(DDF_sitk, sitk.sitkVectorFloat64)
    DDF_sitk_tfm= sitk.DisplacementFieldTransform(DDF_sitk)
    
    return DDF_sitk_tfm

def replace_all(input_arr, k=None, v=None):
    '''
        Efficiently replaces all elements of integer input array using the mapping k->v
        There are several caveats:
         - The input array and the keys array must contain integers
         - The memory requirements are proportional to the largest element of k
         
        Parameters
        ----------
        input_arr: ndarray
            Array of integers whose values will be replaced
        k: array or None
            1D integer array with the keys for the substitution
            If None, use: [0,...,max(input_arr)]
        v: array or None
            1D array with the values for the substitution
            If None, use: [0,...,len(k)]
            
        Return
        ------
        out: array
    '''
    if k is None: k= np.arange(input_arr.max())
    if v is None: v= np.arange(len(k))
    shape= input_arr.shape
    mapping_arr = np.zeros(k.max()+1, dtype=v.dtype)
    mapping_arr[k] = v
    out = mapping_arr[input_arr.flatten()].reshape(shape)
    return out
    
def get_surface_mesh(mesh):
    '''
        Takes a meshio volumetric mesh (tetra + tri elements) and returns
        original point indices, original point positions, and the surface mesh
    '''
    tri_idx= mesh.cells_dict['triangle']
    point_idx= np.unique(tri_idx.flatten()) #old_point_idx
    point_pos= mesh.points[point_idx].astype(np.float32)
    tri_idx_new= replace_all(tri_idx, k=point_idx)
    
#     surf_mesh = pv.PolyData(point_pos, tri_idx_new)
    celltypes= np.empty(tri_idx_new.shape[0], dtype=np.uint8)
    celltypes[:]= vtk.VTK_TRIANGLE
    cells= np.concatenate([np.zeros((tri_idx_new.shape[0], 1)) + 3, 
                            tri_idx_new], axis=1).ravel().astype(np.uint32)
    surf_mesh= UnstructuredGrid(cells, celltypes, point_pos)
    
    return point_idx, point_pos, surf_mesh

def quality(pv_mesh, measures=['min_angle', 'scaled_jacobian', 'aspect_ratio'], plot=True):
    '''
        Compute mesh quality measurements (and possibly plot the historgrams)
        for a pyvista mesh
    '''
    print(f'Quality results:')
    qualities=[]
    for measure in measures:
        qual= pv_mesh.compute_cell_quality(quality_measure=measure) #min_angle, scaled_jacobian, squish?
        values= qual.cell_data['CellQuality']
        if plot:
            plt.figure()
            sns.distplot(values)
            plt.title('Quality (%s)'%measure)
            plt.text(values.min(), 0.001, '%.4f'%values.min(), color='r')
            plt.text(values.max(), 0.001, '%.4f'%values.max(), color='r')
        m_min, m_median, m_max= float(values.min()), float(np.median(values)), float(values.max())
        print(f'- {measure}: min: {m_min:.4f}, median: {m_median:.4f}, max: {m_max:.4f}')
        qualities.append([m_min, m_median, m_max])
    return np.array(qualities)

#Others
class PlotCallback():
    'Callback to pass to CPD for plotting point set registration evolution'
    def __init__(self, points_moving, points_fixed=None, to_numpy= lambda a:a, hide_axis= False, size=[8, 8],
                 plot_interval=1, axis=[0,2,1], mode='3d', moving_color='r', fixed_color='b', save_as=None,
                 moving_last_N=0, fixed_last_N=0, invert_yaxis=True, 
                 x_limits=[0,80], y_limits=[0,80], z_limits=[0,80]):
        self.points_moving= points_moving
        self.points_fixed= points_fixed
        self.axis= axis[:2] if mode == '2d' else axis
        self.mode= mode
        self.plot_interval= plot_interval
        self.moving_color= moving_color
        self.fixed_color= fixed_color
        self.save_as= save_as
        self.to_numpy= to_numpy
        self.hide_axis= hide_axis
        self.size= size
        self.x_limits= x_limits
        self.y_limits= y_limits
        self.z_limits= z_limits
        self.moving_last_N= moving_last_N
        self.fixed_last_N= fixed_last_N
        self.invert_yaxis= invert_yaxis
        
        self.count= 0
        self.lbl= ['x', 'y', 'z']
        
        self.first_style= dict(marker='.', linestyle='None', markersize=1)
        self.last_style= dict(marker='.', linestyle='-', markersize=1)
        
    def __call__(self, tf_param):
        if not (self.count % self.plot_interval):
            self._plot(tf_param)
        self.count+= 1
        
    def _plot(self, tf_param):
        #Transform points and convert to numpy
        if self.points_fixed is not None: points_fixed= self.to_numpy(self.points_fixed)
        points_moving=self.to_numpy(tf_param.transform(self.points_moving))

        #Separate first points (prostate) from last points (urethra)
        pm_first= points_moving[:-self.moving_last_N] if self.moving_last_N != 0 else points_moving
        pm_last= points_moving[-self.moving_last_N:] if self.moving_last_N != 0 else None
        pf_first= points_fixed[:-self.fixed_last_N] if self.fixed_last_N != 0 else points_fixed
        pf_last= points_fixed[-self.fixed_last_N:] if self.fixed_last_N != 0 else None

        #Set up figure
        plt.figure(figsize=self.size)
        if self.mode=='2d': ax= plt.axes()
        elif self.mode=='3d': ax= plt.axes(projection='3d')
        else: raise RuntimeError(f'Unrecognized plot mode: {self.mode}')

        #Plot
        ax.plot(*[pm_first[:,i] for i in self.axis], color=self.moving_color, **self.first_style)
        if pm_last is not None:
            ax.plot(*[pm_last[:,i] for i in self.axis], color=self.moving_color, **self.last_style)
        if points_fixed is not None: 
            ax.plot(*[pf_first[:,i] for i in self.axis], color=self.fixed_color, **self.first_style)
            if pf_last is not None:
                ax.plot(*[pf_last[:,i] for i in self.axis], color=self.fixed_color, **self.last_style)

        #Decorate and configure style
        ax.set_xlabel(self.lbl[self.axis[0]]); ax.set_ylabel(self.lbl[self.axis[1]])
        if self.mode=='3d': ax.set_zlabel(self.lbl[self.axis[2]])
        ax.set_xlim(self.x_limits); ax.set_ylim(self.y_limits)
        if self.mode=='3d': ax.set_zlim(self.z_limits)
        if self.invert_yaxis: ax.invert_yaxis()
        if self.hide_axis: plt.axis('off')
        plt.tight_layout()
        plt.show()

        #Save
        if self.save_as is not None: plt.savefig(self.save_as, dpi=300)