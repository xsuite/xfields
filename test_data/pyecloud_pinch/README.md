# example to run e-cloud with xsuite


## First do the pinch

```
cd Quadrupole
python 000_buildup.py
python 001_pinch.py
```

## "Refine" the pinch

Dependencies:
```
git clone git@github.com:ecloud-cern/refine_pinch.git
pip install git+https://github.com/kparasch/TricubicInterpolation.git@v1.1.0
```

To run the refinement:
```
cd refine_pinch.py
cp ../Pinch.h5 .
python reorder_slices.py Pinch.h5
python refine_pinch.py Pinch.h5 --DTO 2
```

output refined pinch that is ready to be used in xsuite will be called:
 ```Quadrupole/refine_pinch/refined_Pinch_MTI1.0_MLI1.0_DTO2.0_DLO1.0.h5```

In the refinement code:
- MTI (magnify tranverse input) refers to ratio between input grid and auxilliary grid
- MLI (magnify longitudinal input) refers to ratio between number of slices in the input grid and and the auxilliary grid
- DTO (demagnify transverse output) refers to ratio between input grid and output grid
- DLO (demagnify longitudinal output) refers to ratio between number of slices in the input grid and and the output grid
