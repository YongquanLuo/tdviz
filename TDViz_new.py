# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:54:13 2024

@author: Luo
"""

from traits.api import HasTraits, Button, Instance, List, Str, Enum, Float, File, Int, ReadOnly
from traitsui.api import View, Item, VGroup, HSplit, HGroup, FileEditor
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi import mlab
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.io.fits as fits
import numpy as np
import os, glob
from spectral_cube import SpectralCube as sc
from astropy.wcs import WCS

class TDViz(HasTraits):
    fitsfile    = File(filter=[u"*.fits"])
    plotbutton1 = Button(u"Plot")
    plotbutton2 = Button(u"Plot")
    plotbutton3 = Button(u"Plot")
    clearbutton = Button(u"Clear")
    scene = Instance(MlabSceneModel, ())
    rendering = Enum("Surface-Spectrum", "Surface-Intensity", "Volume-Intensity")
    save_the_scene = Button(u"Save")
    save_in_file = Str("test.x3d")
    movie = Button(u"Movie")
    iteration	= Int(0)
    quality	= Int(8)
    delay	= Int(0)
    angle	= Int(360)
    spin = Button(u"Spin")
    #zscale = Float(1.0)
    zscale = Int(1)
    xstart = Int(0)
    xend   = Int(1)
    ystart = Int(0)
    yend   = Int(1)
    # zstart = Int(0)
    # zend   = Int(1)
    datamin= Float(0.0)
    datamax= Float(1.0)
    opacity= Float(0.4)
    dist = Float(0.0)
    leng = Float(0.0)
    vsp = Float(0.0)
    contfile = File(filter=[u"*.fits"])
    contour_cmap = Enum("jet", "gist_ncar", "viridis", "cubehelix", "plasma", "RdGy", "RdYlGn", "magma", "spring", "summer", "winter", "gist_rainbow")
    rms = Float(0.0)
    Vlsr = ''
    
    v1 = Float(0.0)
    v2 = Float(0.0)
    v_range = list([0,0])
    
    
    # contour3d setting
    vmin = Float(0.0)
    vmax = Float(0.0)

    view = View(
        HSplit(
            VGroup(
                Item("fitsfile", label=u"Select a FITS datacube", show_label=True, editor=FileEditor(dialog_style='open')),
                Item("rendering", tooltip=u"Choose the rendering type", show_label=True),
                Item('plotbutton1', tooltip=u"Plot 3D surfaces, color coded by velocities", visible_when="rendering=='Surface-Spectrum'"),
                Item('plotbutton2', tooltip=u"Plot 3D surfaces, color coded by intensities", visible_when="rendering=='Surface-Intensity'"),
                Item('plotbutton3', tooltip=u"Plot 3D dots, color coded by intensities", visible_when="rendering=='Volume-Intensity'"),
                "clearbutton",
                HGroup(Item('xstart', tooltip=u"starting pixel in X axis", show_label=True, springy=True),
                	     Item('xend', tooltip=u"ending pixel in X axis", show_label=True, springy=True)
                       ),
                HGroup(Item('ystart', tooltip=u"starting pixel in Y axis", show_label=True, springy=True),
                	     Item('yend', tooltip=u"ending pixel in Y axis", show_label=True, springy=True)
                       ),
                HGroup(Item('v1', tooltip=u"starting velocity(km/s) in spectral axis", show_label=True, springy=True),
                	     Item('v2', tooltip=u"ending velocity(km/s) in spectral axis", show_label=True, springy=True)
                       ),
                HGroup(Item('datamax', tooltip=u"Maximum datapoint shown", show_label=True, springy=True),
                	     Item('datamin', tooltip=u"Minimum datapoint shown", show_label=True, springy=True)
                       ),
                HGroup(Item('dist', tooltip=u"Put a distance in kpc", show_label=True),
                	     Item('leng', tooltip=u"Put a non-zero bar length in pc to show the scale bar", show_label=True),
                	     Item('vsp', tooltip=u"Put a non-zero velocity range in km/s to show the scale bar", show_label=True)
                       ),
                Item('Vlsr', label='Vlsr', tooltip=u"systematic velocity in km/s", show_label=True),

                HGroup(Item('zscale', tooltip=u"Stretch the datacube in Z axis", show_label=True),
                	     Item('opacity', tooltip=u"Opacity of the scene", show_label=True), 
                       Item('rms', label="rms", style='readonly', show_label=True, springy=True),
                       show_labels=False),
                HGroup(Item("contour_cmap", label='colormap', tooltip=u"Choose a colormap for the 3Dcontour",show_label=True),
                       Item('vmax', show_label=True, springy=True),
                       Item('vmin', show_label=True, springy=True)
                       ),
                Item('_'),
                Item("contfile", label=u"Add background contours", tooltip=u"This file must be of the same (first two) dimension as the datacube!!!", show_label=True, editor=FileEditor(dialog_style='open')),
                Item('_'),
                HGroup(Item("spin", tooltip=u"Spin 360 degrees", show_label=False),
                	     Item("movie", tooltip="Make a GIF movie", show_label=False)
                       ),
                HGroup(Item('iteration', tooltip=u"number of iterations, 0 means inf.", show_label=True),
                	     Item('quality', tooltip=u"quality of plots, 0 is worst, 8 is good.", show_label=True)
                       ),
                HGroup(Item('delay', tooltip=u"time delay between frames, in millisecond.", show_label=True),
                	     Item('angle', tooltip=u"angle the cube spins", show_label=True)
                       ),
                Item('_'),
                HGroup(Item("save_the_scene", tooltip=u"Save current scene in a 3D model file"),
                	     Item("save_in_file", tooltip=u"3D model file name", show_label=False), visible_when="rendering=='Surface-Spectrum' or rendering=='Surface-Intensity'"
                       ),
                show_labels=False
            ),
            VGroup(
                Item(name='scene',
                    editor=SceneEditor(scene_class=MayaviScene),
                    resizable=True,
                    height=600,
                    width=900
                    ), show_labels=False
            )
        ),
    resizable=True,
    title=u"TDViz"
    )
    
    def _fitsfile_changed(self):
        
        self.scu_a = sc.read(self.fitsfile).with_spectral_unit(u.km/u.s, velocity_convention='radio')
        dat = self.scu_a.hdu.data
        self.hdr = self.scu_a.header
        self.cube_shape = np.shape(dat)
        self.v_range[0] = np.nanmin(self.scu_a.spectral_axis.to(u.km/u.s).value)
        self.v_range[1] = np.nanmax(self.scu_a.spectral_axis.to(u.km/u.s).value)
        
        naxis = self.hdr['NAXIS']
        if naxis == 4:
            self.data = np.swapaxes(dat[0],0,2)
        elif naxis == 3:
            self.data = np.swapaxes(dat,0,2)
        
        self.data[np.isinf(self.data)] = np.nan
    
        self.datamax = np.nanmax(self.data)
        self.datamin = np.nanmin(self.data)
        
        self.vmin = self.datamin
        self.vmax = self.datamax
        
        
        self.xend    = self.data.shape[0] - 1 
        self.yend    = self.data.shape[1] - 1
        self.zend    = self.data.shape[2] - 1
        
        rms = np.nanstd(self.data, axis = (0,1))
        self.rms = np.nanmean(rms)
        
        
        
        self.v1 = self.v_range[0]
        self.v2 = self.v_range[1]
    
    def loaddata(self):
    
        if self.xstart < 0:
            print('Wrong number!')
            self.xstart = 0
        if self.xend > self.cube_shape[2]-1:
            print('Wrong number!')
            self.xend = self.cube_shape[2]-1
        if self.ystart < 0:
            print('Wrong number!')
            self.ystart = 0
        if self.yend > self.cube_shape[1]-1:
            print('Wrong number!')
            self.yend = self.cube_shape[1]-1
        
        if ((self.v1 < self.v_range[0]) | (self.v1 > self.v_range[1])):
            print('Wrong number!')
            self.v1 = self.v_range[0]
        if ((self.v2 > self.v_range[1]) | (self.v2 < self.v_range[0])):
            print('Wrong number!')
            self.v2 = self.v_range[2]

        self.scu = self.scu_a.spectral_slab(self.v1 * u.km/u.s, self.v2 * u.km/u.s)[:,self.ystart:self.yend+1,self.xstart:self.xend+1]
        self.spectral_axis = self.scu.spectral_axis
        dat = self.scu.hdu.data
        self.hdr = self.scu.header
        naxis = self.hdr['NAXIS']
        if naxis == 4:
            self.data = np.swapaxes(dat[0],0,2)
        elif naxis == 3:
            self.data = np.swapaxes(dat,0,2)
        self.data[np.isinf(self.data)] = np.nan
        


        region = np.nan_to_num(self.data)
        
        from scipy.interpolate import splrep
        from scipy.interpolate import splev
        
        vol=region.shape
        stretch=self.zscale
        sregion=np.empty((vol[0],vol[1],vol[2]*stretch))
        chanindex=np.linspace(0,vol[2]-1,vol[2])
        chanindex2=np.linspace(0,vol[2]-1,vol[2]*stretch)
        for j in range(0,vol[0]):
            for k in range(0,vol[1]):
                spec=region[j,k,:]
                tck=splrep(chanindex,spec,k=1)
                sregion[j,k,:]=splev(chanindex2,tck)
        self.sregion = sregion
        
        unit = self.spectral_axis.unit
        tck = splrep(chanindex, self.spectral_axis.value, k = 1)
        self.spectral_axis = splev(chanindex2, tck) * unit

        if self.datamin < np.min(self.sregion):
            print('Wrong number!')
            self.datamin = np.min(self.sregion)
        if self.datamax > np.max(self.sregion):
            print('Wrong number!')
            self.datamax = np.max(self.sregion)
        self.xrang = region.shape[0]
        self.yrang = region.shape[1]
        self.zrang = region.shape[2] * stretch
        
        ## Keep a record of the coordinates:
        crval1 = self.hdr['crval1']
        cdelt1 = self.hdr['cdelt1']
        crpix1 = self.hdr['crpix1']
        crval2 = self.hdr['crval2']
        cdelt2 = self.hdr['cdelt2']
        crpix2 = self.hdr['crpix2']
        crval3 = self.hdr['crval3']
        cdelt3 = self.hdr['cdelt3']
        crpix3 = self.hdr['crpix3']
        
        ra_start = (0 + 1 - crpix1) * cdelt1 + crval1
        ra_end = (self.xrang + 1  - crpix1) * cdelt1 + crval1
        
        dec_start = (0 + 1 - crpix2) * cdelt2 + crval2
        dec_end = (self.yrang + 1 - crpix2) * cdelt2 + crval2
        
        # vel_start = (0 +1 - crpix3) * cdelt3 + crval3
        # vel_end = (region.shape[2] + 1 - crpix3) * cdelt3 + crval3
        vel_start = np.nanmin(self.scu.spectral_axis.to(u.km/u.s).value)
        vel_end = np.nanmax(self.scu.spectral_axis.to(u.km/u.s).value)
        # vel_start = self.v1
        # vel_end = self.v2
        
        print('original cdelt3=', self.hdr['cdelt3'])
        
        ## Flip the V axis
        if cdelt3 > 0:
            self.sregion = self.sregion[:,:,::-1]
            self.hdr['crpix3'] = region.shape[2] - crpix3
            self.hdr['cdelt3'] = -cdelt3
            
        
        
        self.extent =[ra_start, ra_end, dec_start, dec_end, vel_start, vel_end]
        
        print("ra_start =", ra_start)
        print("ra_end =", ra_end)
        print("dec_start =", dec_start)
        print("dec_end =", dec_end)
        print("vel_start =", vel_start)
        print("vel_end =", vel_end)


    def labels(self):
        
        fontsize = max(self.xrang, self.yrang)/40.
        tcolor = (1,1,1)
        mlab.text3d(self.xrang/2,-10,self.zrang+10,'R.A.',scale=fontsize,orient_to_camera=True,color=tcolor)
        mlab.text3d(-10,self.yrang/2,self.zrang+10,'Decl.',scale=fontsize,orient_to_camera=True,color=tcolor)
        mlab.text3d(-10,-10,self.zrang/2-10,'V (km/s)',scale=fontsize,orient_to_camera=True,color=tcolor)

        mlab.text3d(0, self.yrang + self.yrang/5, self.zrang+10, os.path.basename(self.fitsfile), scale=fontsize, orient_to_camera=False)

        # Add scale bars
        if self.leng != 0.0:
            distance = self.dist * 1e3
            length = self.leng
            leng_pix = np.round(length/distance/np.pi*180./np.abs(self.hdr['cdelt1']))
            bar_x = [self.xrang-20-leng_pix, self.xrang-20]
            bar_y = [self.yrang-10, self.yrang-10]
            bar_z = [0, 0]
            mlab.plot3d(bar_x, bar_y, bar_z, color=tcolor, tube_radius=1.)
            mlab.text3d(self.xrang-30-leng_pix,self.yrang-25,0,'{:.2f} pc'.format(length),scale=fontsize,orient_to_camera=False,color=tcolor)
        
        if self.vsp != 0.0:
            vspan = self.vsp
            # vspan_pix = np.round(vspan/np.abs(self.hdr['cdelt3']/1e3))
            vspan_pix = np.round(vspan/np.abs(self.hdr['cdelt3']))
            bar_x = [self.xrang, self.xrang]
            bar_y = [self.yrang-10, self.yrang-10]
            bar_z = np.array([5, 5+vspan_pix])*self.zscale
            mlab.plot3d(bar_x, bar_y, bar_z, color=tcolor, tube_radius=1.)
            mlab.text3d(self.xrang,self.yrang-25,45,'{:.1f} km/s'.format(vspan),scale=fontsize,orient_to_camera=False,color=tcolor,orientation=(0,90,0))

        # Label the coordinates of the corners
        # Lower left corner
        ra0 = self.extent[0]; dec0 = self.extent[2]
        c = SkyCoord(ra=ra0*u.degree, dec=dec0*u.degree, frame='icrs')
        RA_ll = str(int(c.ra.hms.h))+'h'+str(int(c.ra.hms.m))+'m'+str(round(c.ra.hms.s,1))+'s'
        mlab.text3d(0,-10,self.zrang+5,RA_ll,scale=fontsize,orient_to_camera=True,color=tcolor)
        DEC_ll = str(int(c.dec.dms.d))+'d'+str(int(abs(c.dec.dms.m)))+'m'+str(round(abs(c.dec.dms.s),1))+'s'
        mlab.text3d(-40,0,self.zrang+5,DEC_ll,scale=fontsize,orient_to_camera=True,color=tcolor)
        # Upper right corner
        ra0 = self.extent[1]; dec0 = self.extent[3]
        c = SkyCoord(ra=ra0*u.degree, dec=dec0*u.degree, frame='icrs')
        RA_ll = str(int(c.ra.hms.h))+'h'+str(int(c.ra.hms.m))+'m'+str(round(c.ra.hms.s,1))+'s'
        mlab.text3d(self.xrang,-10,self.zrang+5,RA_ll,scale=fontsize,orient_to_camera=True,color=tcolor)
        DEC_ll = str(int(c.dec.dms.d))+'d'+str(int(abs(c.dec.dms.m)))+'m'+str(round(abs(c.dec.dms.s),1))+'s'
        mlab.text3d(-40,self.yrang,self.zrang+5,DEC_ll,scale=fontsize,orient_to_camera=True,color=tcolor)

        # V axis
        if self.extent[5] > self.extent[4]:
            v0 = self.extent[4]; v1 = self.extent[5]
        else:
            v0 = self.extent[5]; v1 = self.extent[4]
        mlab.text3d(-10,-10,self.zrang,str(round(v0,1)),scale=fontsize,orient_to_camera=True,color=tcolor)
        mlab.text3d(-10,-10,0,str(round(v1,1)),scale=fontsize,orient_to_camera=True,color=tcolor)
        
        mlab.axes(self.field, ranges=self.extent, x_axis_visibility=False, y_axis_visibility=False, z_axis_visibility=False)
        
        mlab.outline()


    def _plotbutton1_fired(self):
        mlab.clf()
        self.loaddata()
        
        
        
        # self.sregion[np.where(self.sregion<self.datamin)] = self.datamin
        # self.sregion[np.where(self.sregion>self.datamax)] = self.datamax
        # self.sregion[np.where(self.sregion<self.datamin)] = 0
        # self.sregion[np.where(self.sregion>self.datamax)] = 0
        
        region = self.sregion
        region[np.where(region < self.datamin)] = self.datamin
        region[np.where(region > self.datamax)] = self.datamax
        
        # The following codes from: http://docs.enthought.com/mayavi/mayavi/auto/example_atomic_orbital.html#example-atomic-orbital
        field = mlab.pipeline.scalar_field(region)     # Generate a scalar field
        # field = mlab.pipeline.scalar_field(region, vmin = self.datamax, vmax = self.datamax)

    
        colored = region
        
        vol=self.sregion.shape
        for v in range(0,vol[2]):
            # colored[:,:,v] = self.extent[4] + v*(-1)*abs(self.hdr['cdelt3'])
            colored[:,:,v] = self.extent[4] - v * self.hdr['cdelt3']
            # for i in range(vol[0]):
            #     for j in range(vol[1]):
            #         if (region[i, j, v] != np.nan):
            #             colored[i,j,v] = self.extent[4] - v * self.hdr['cdelt3']
        # colored = colored[:,:,:]
        self.colored = colored
        
        field.image_data.point_data.add_array(colored.T[::-1,:,:].ravel())
        field.image_data.point_data.get_array(1).name = 'color'
        field.update()
        
        field2 = mlab.pipeline.set_active_attribute(field, point_scalars='scalar')
        
        contour = mlab.pipeline.contour(field2)
        
        # contour.contour.maximum_contour = self.datamax
        # contour.contour.minimum_contour = self.datamin
        
        contour2 = mlab.pipeline.set_active_attribute(contour, point_scalars='color')
        
        # contour2.contour.maximum_contour = self.datamax
        # contour2.contour.minimum_contour = self.datamin
        
        mlab.pipeline.surface(contour2, colormap='jet', opacity=self.opacity)
        
        self.field = field2
        # self.field.scene.render()
        self.labels()
        
        
        
        ##############################################
        
        # Tb_cube = np.swapaxes(self.sregion,0,2)
        # nv, ny, nx = np.shape(Tb_cube)
        # v_cube = np.zeros((nv, ny, nx))
        # Tb_int = np.nansum(Tb_cube,axis=0)
        
        # sp_v = self.spectral_axis.value
        # for i in np.arange(nv):
        #     v_cube[i, :, :] = sp_v[i]
        
        # max0 = np.nanmean(Tb_cube, axis = 0)
        
        # src = mlab.pipeline.scalar_field(Tb_cube/max0)
        # # velocity as the color scale:
        # src.image_data.point_data.add_array(v_cube.T.ravel())
        # src.image_data.point_data.get_array(1).name = 'vsys'
        # # Make sure that the dataset is up to date with the different arrays:
        # src.update()
        # # We select the 'scalar' attribute, ie the norm of v_cube
        # src2 = mlab.pipeline.set_active_attribute(src, point_scalars='scalar')
        
        
        # # ------contour surface-------
        # contour = mlab.pipeline.contour(src2)
        # contour.filter.contours=[0.1, 0.3, 0.5, 0.7, 0.9]
        # contour2 = mlab.pipeline.set_active_attribute(contour,point_scalars='vsys')
        # mlab.pipeline.surface(contour2, colormap='jet', opacity=0.5, extent=[0,nv,0,ny,0,nx])
        # # extent = [0,nx/2,0,nx,0,ny]
        # # extent = [0,1,0,3,0,4]
        # # extent = [9.1,12,0,199,0,206]
        # mlab.colorbar(title='Vsys', orientation='vertical', nb_labels=3)
        # # mlab.axes(contour2, xlabel='Vsys', ylabel='Ra', zlabel='Dec')
        # mlab.outline(extent=[0,nv,0,ny,0,nx])  #boxs
        # # mlab.surf(Tb_int)
        # #mlab.contour_surf(xm[0,:],ym[:,0],np.transpose(Tb_int,(1,0)),
        # #                  contours=[0.3,0.8,1.3,1.8],color=(0,0,0),warp_scale=0)
        
        # mlab.contour_surf(Tb_int/Tb_int.max(), contours=[0.3,0.5,0.7,0.9],color=(0,0,0),warp_scale=0)
        # #ranges = [0,100,0,200,0,200]
        # #mlab.gcf().scene.background = (0.6, 0.6, 0.9)
        
        # self.labels()
        
        
        
        
        
        
        
        
        
        
        
        
        
        if self.Vlsr != '':
            vol = self.sregion.shape
            Vlsr_pix = self.hdr['crpix3'] - (self.hdr['crval3'] - float(self.Vlsr)) / self.hdr['cdelt3']
            z = np.ones((vol[0], vol[1])) * Vlsr_pix * self.zscale
            surf = mlab.surf(np.arange(vol[0]), np.arange(vol[1]), z, color = (0.6,0.6,0.6))
            surf.actor.property.opacity = 0.5

        print('In plotting, cdelt3 =', self.hdr['cdelt3'])
        print("#####")

        if self.contfile != '':
            im = fits.open(self.contfile)
            dat = im[0].data
            channel = dat
            region = np.swapaxes(channel[self.ystart:self.yend,self.xstart:self.xend],0,1) * 1.0
            contour_levels = (np.arange(0.2, 1, 0.2) * np.nanmax(region)).tolist()
            field = mlab.contour3d(region, colormap='gist_ncar', contours = contour_levels)
        
        mlab.view(azimuth=0, elevation=0, distance='auto')
        mlab.show()






    def _plotbutton2_fired(self):
        mlab.clf()
        self.loaddata()
        field=mlab.contour3d(self.sregion, colormap = self.contour_cmap)     # Generate a scalar field
        # field=mlab.contour3d(self.sregion, colormap = 'gist_ncar')     # Generate a scalar field
        field.module_manager.scalar_lut_manager.data_range = [self.vmin, self.vmax]
        field.contour.maximum_contour = self.datamax
        field.contour.minimum_contour = self.datamin
        field.actor.property.opacity = self.opacity
        
        self.field = field
        self.labels()
        
        if self.Vlsr != '':
            
            vol = self.sregion.shape
            Vlsr_pix = self.hdr['crpix3'] - (self.hdr['crval3'] - float(self.Vlsr)) / self.hdr['cdelt3']
            z = np.ones((vol[0], vol[1])) * Vlsr_pix * self.zscale
            surf = mlab.surf(np.arange(vol[0]), np.arange(vol[1]), z, color = (0.6,0.6,0.6))
            surf.actor.property.opacity = 0.5

        print('In plotting, cdelt3 =', self.hdr['cdelt3'])
        print("#####")

        if self.contfile != '':
            im = fits.open(self.contfile)
            dat = im[0].data
            # dat[np.isnan(dat)] = 0
            # dat[dat <= 0] = 0
            channel = dat
            region = np.swapaxes(channel[self.ystart:self.yend,self.xstart:self.xend],0,1) * 1.0
            contour_levels = (np.arange(0.2, 1, 0.2) * np.nanmax(region)).tolist()
            field = mlab.contour3d(region, colormap='gist_ncar', contours = contour_levels)
            # field.contour.minimum_contour = 5

        mlab.view(azimuth=0, elevation=0, distance='auto')
        mlab.show()

    def _plotbutton3_fired(self):
        mlab.clf()
        self.loaddata()
        field = mlab.pipeline.scalar_field(self.sregion) # Generate a scalar field
        mlab.pipeline.volume(field,vmax=self.vmax,vmin=self.vmin)
        
        self.field = field
        self.labels()
        mlab.view(azimuth=0, elevation=0, distance='auto')
        mlab.colorbar()
        mlab.show()


    def _clearbutton_fired(self):
        mlab.clf()

app = TDViz()
app.configure_traits() 