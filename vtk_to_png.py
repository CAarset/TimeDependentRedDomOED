#!/usr/bin/env python

from matplotlib.pyplot import get_cmap

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkIOImage import (
    vtkBMPWriter,
    vtkJPEGWriter,
    vtkPNGWriter,
    vtkPNMWriter,
    vtkPostScriptWriter,
    vtkTIFFWriter
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkWindowToImageFilter
)

from vtk import vtkUnstructuredGridReader
from vtk import vtkDataSetMapper

from os import remove

def file_to_vtk(filename, scalar_range = None):
    # Read the source file.
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()
    output_port = reader.GetOutputPort()
    if scalar_range is None:
        scalar_range = output.GetScalarRange()
    
    # Create the mapper that corresponds the objects of the vtk file
    # into graphics elements
    mapper = vtkDataSetMapper()
    mapper.SetInputConnection(output_port)
    
    mapper.SetScalarRange(scalar_range)

    # Create colormap
    #cmap = get_cmap("plasma")#("coolwarm")#
    lut = vtkLookupTable()
    #lut.SetNumberOfTableValues(len(cmap.colors))
    #for i,val in enumerate(cmap.colors):
    #    lut.SetTableValue(i,val[0],val[1],val[2])
    lut.SetNumberOfColors(256)
    lut.SetHueRange(0.667, 0) #0.667
    lut.SetValueRange(1,0.9)
    #lut.Build()

    mapper.SetLookupTable(lut)

    # Create the Actor
    actor = vtkActor()
    actor.SetMapper(mapper)
    
    # Create the Renderer
    renderer = vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1) # Set background to white

    # Create the RendererWindow
    #renderer_window = vtkRenderWindow()
    #renderer_window.AddRenderer(renderer)

    # Create the RendererWindowInteractor and display the vtk_file
    #interactor = vtkRenderWindowInteractor()
    #interactor.SetRenderWindow(renderer_window)
    #interactor.Initialize()
    #interactor.Start()
    return renderer


def vtk_to_png(filename, imagename, delete = False, scalar_range = None):
    # create source
    ren = file_to_vtk(filename, scalar_range = scalar_range)

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(512*4, 512*4)

    renWin.SetWindowName('ImageWriter')
    renWin.Render()

    WriteImage(imagename, renWin)#, rgba=False)

    if delete:
        remove(filename)

    #iren.Initialize()
    #iren.Start()


def WriteImage(filename, renWin, rgba=True):
    '''
    Write the render window view to an image file.

    Image types supported are:
     BMP, JPEG, PNM, PNG, PostScript, TIFF.
    The default parameters are used for all writers, change as needed.

    :param fileName: The file name, if no extension then PNG is assumed.
    :param renWin: The render window.
    :param rgba: Used to set the buffer type.
    :return:
    '''

    import os

    if filename:
        # Select the writer to use.
        path, ext = os.path.splitext(filename)
        ext = ext.lower()
        if not ext:
            ext = '.png'
            filename = filename + ext
        if ext == '.bmp':
            writer = vtkBMPWriter()
        elif ext == '.jpg':
            writer = vtkJPEGWriter()
        elif ext == '.pnm':
            writer = vtkPNMWriter()
        elif ext == '.ps':
            if rgba:
                rgba = False
            writer = vtkPostScriptWriter()
        elif ext == '.tiff':
            writer = vtkTIFFWriter()
        else:
            writer = vtkPNGWriter()

        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(renWin)
        windowto_image_filter.SetScale(1)  # image quality
        if rgba:
            windowto_image_filter.SetInputBufferTypeToRGBA()
        else:
            windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            windowto_image_filter.ReadFrontBufferOff()
            windowto_image_filter.Update()

        writer.SetFileName(filename)
        writer.SetInputConnection(windowto_image_filter.GetOutputPort())
        writer.Write()
    else:
        raise RuntimeError('Need a filename.')


if __name__ == '__main__':
    main()