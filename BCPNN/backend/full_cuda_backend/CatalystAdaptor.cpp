#include "CatalystAdaptor.h"

#include <iostream>

#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkCPPythonScriptPipeline.h>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkUnsignedCharArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkImageData.h>

namespace
{
vtkCPProcessor* Processor = nullptr;
vtkImageData* VTKGrid = nullptr;
const char* InputName = "wmask";

size_t _hypercolumns, _rows, _columns;

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void UpdateVTKAttributes(vtkCPInputDataDescription* idd, unsigned int timeStep, uint8_t *wmask)
{
  // Get a reference to the grid's point data object.
  vtkPointData* vtk_point_data = VTKGrid->GetPointData();

  // Array of grid's dimensions
  int* dims = VTKGrid->GetDimensions();

  for (size_t h = 0; h < _hypercolumns; h++) {
    // electric charge density for species 0 and 1
    vtkNew<vtkUnsignedCharArray> wmask_array{};
    wmask_array->SetName(std::to_string(h).c_str());
    wmask_array->SetNumberOfComponents(1);
    wmask_array->SetNumberOfTuples(VTKGrid->GetNumberOfPoints());
    vtk_point_data->AddArray(wmask_array);

    long p = 0;
    for (size_t i = 0; i < _rows; ++i) {
      for (size_t j = 0; j < _columns; ++j) {
        wmask_array->SetValue(p, wmask[(i * _columns + j) * _hypercolumns + h]);
	p++;
      }
    }
  }
}

//----------------------------------------------------------------------------
void BuildVTKDataStructures(vtkCPInputDataDescription *idd, unsigned int timeStep, uint8_t *wmask)
{
  // feed data to grid
  UpdateVTKAttributes(idd, timeStep, wmask);
}
}

namespace Adaptor
{

//----------------------------------------------------------------------------
void Initialize(const char* script, const size_t rows, const size_t columns, const size_t hypercolumns)
{
  if (Processor == NULL)
  {
    Processor = vtkCPProcessor::New();
    Processor->Initialize();
  }
  else
  {
    Processor->RemoveAllPipelines();
  }
  vtkNew<vtkCPPythonScriptPipeline> pipeline;
  pipeline->Initialize(script);
  Processor->AddPipeline(pipeline);

  _rows = rows;
  _columns = columns;
  _hypercolumns = hypercolumns;

  if (VTKGrid == NULL)
  {
    // The grid structure isn't changing so we only build it
    // the first time it's needed. If we needed the memory
    // we could delete it and rebuild as necessary.
    VTKGrid = vtkImageData::New();
    VTKGrid->SetDimensions(_rows, _columns, 1);
    VTKGrid->SetOrigin(0,0,0);
    VTKGrid->SetSpacing(1,1,1);
  }
}

//----------------------------------------------------------------------------
void Finalize()
{
  if (Processor)
  {
    Processor->Delete();
    Processor = NULL;
  }
  if (VTKGrid)
  {
    VTKGrid->Delete();
    VTKGrid = NULL;
  }
}

//----------------------------------------------------------------------------
void CoProcess(double time, unsigned int timeStep, uint8_t *wmask)
{
  vtkNew<vtkCPDataDescription> dataDescription;
  dataDescription->AddInput(InputName);
  dataDescription->SetTimeData(time, timeStep);

  if (Processor->RequestDataDescription(dataDescription) != 0)
  {
    vtkCPInputDataDescription* idd = dataDescription->GetInputDescriptionByName(InputName);
    BuildVTKDataStructures(idd, timeStep, wmask);
    idd->SetGrid(VTKGrid);
    Processor->CoProcess(dataDescription);
  }
}
} // end of Catalyst namespace
