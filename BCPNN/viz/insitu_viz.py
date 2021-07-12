import vtk
from vtk.util import numpy_support
from paraview.modules import vtkPVCatalyst as catalyst
from . import image as coprocessor


def coprocess(data, n_hypercolumns, epoch):
    image = vtk.vtkImageData()
    image.SetDimensions(data[0].shape)
    image.SetSpacing([1] * len(data[0].shape))
    image.SetOrigin([0] * len(data[0].shape))
    
    for i in range(n_hypercolumns):
        depthArray = numpy_support.numpy_to_vtk(data[i].ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        depthArray.SetName(str(i))
        image.GetPointData().AddArray(depthArray)
    
    dataDescription = catalyst.vtkCPDataDescription()
    dataDescription.SetTimeData(epoch, epoch)
    dataDescription.AddInput("wmask")
    
    dataDescription.GetInputDescriptionByName("wmask").SetGrid(image)
    coprocessor.DoCoProcessing(dataDescription)

    #writer = vtk.vtkXMLImageDataWriter()
    #writer.SetFileName("./wmask_"+str(epoch)+".vti")
    #writer.SetInputData(image)
    #writer.Update()
    #writer.Write()
