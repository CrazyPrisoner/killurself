З
М
:
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.10.12v1.10.1-0-g4dcfddc5d1}
t
first_placeholderPlaceholder*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ*
dtype0
u
second_placeholderPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ

weight/initial_valueConst*I
value@B>"0 ЫPZб?сAФpџ?шXЉГо?ь`ЖрП %+8Й?і`ЌќIбП*
dtype0*
_output_shapes

:
z
weight
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
Ђ
weight/AssignAssignweightweight/initial_value*
use_locking(*
T0*
_class
loc:@weight*
validate_shape(*
_output_shapes

:
c
weight/readIdentityweight*
_output_shapes

:*
T0*
_class
loc:@weight
Q
CastCastweight/read*
_output_shapes

:*

DstT0*

SrcT0
W
bias/initial_valueConst*
_output_shapes
: *
valueB
 *L?*
dtype0
h
bias
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 

bias/AssignAssignbiasbias/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
	loc:@bias*
validate_shape(
U
	bias/readIdentitybias*
_output_shapes
: *
T0*
_class
	loc:@bias

MatMulMatMulfirst_placeholderCast*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
V

predictionAddMatMul	bias/read*
T0*'
_output_shapes
:џџџџџџџџџ
\
subSub
predictionsecond_placeholder*
T0*'
_output_shapes
:џџџџџџџџџ
J
Pow/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
H
PowPowsubPow/y*
T0*'
_output_shapes
:џџџџџџџџџ
V
ConstConst*
_output_shapes
:*
valueB"       *
dtype0
T
SumSumPowConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
N
	truediv/yConst*
valueB
 *  ДF*
dtype0*
_output_shapes
: 
C
truedivRealDivSum	truediv/y*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
_
gradients/truediv_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
a
gradients/truediv_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
Р
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
e
gradients/truediv_grad/RealDivRealDivgradients/Fill	truediv/y*
T0*
_output_shapes
: 
­
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
G
gradients/truediv_grad/NegNegSum*
T0*
_output_shapes
: 
s
 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/Neg	truediv/y*
T0*
_output_shapes
: 
y
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1	truediv/y*
T0*
_output_shapes
: 
t
gradients/truediv_grad/mulMulgradients/Fill gradients/truediv_grad/RealDiv_2*
_output_shapes
: *
T0
­
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients/truediv_grad/tuple/group_depsNoOp^gradients/truediv_grad/Reshape!^gradients/truediv_grad/Reshape_1
й
/gradients/truediv_grad/tuple/control_dependencyIdentitygradients/truediv_grad/Reshape(^gradients/truediv_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape*
_output_shapes
: 
п
1gradients/truediv_grad/tuple/control_dependency_1Identity gradients/truediv_grad/Reshape_1(^gradients/truediv_grad/tuple/group_deps*
_output_shapes
: *
T0*3
_class)
'%loc:@gradients/truediv_grad/Reshape_1
q
 gradients/Sum_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
Џ
gradients/Sum_grad/ReshapeReshape/gradients/truediv_grad/tuple/control_dependency gradients/Sum_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
[
gradients/Sum_grad/ShapeShapePow*
_output_shapes
:*
T0*
out_type0

gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
[
gradients/Pow_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
]
gradients/Pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Д
(gradients/Pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Pow_grad/Shapegradients/Pow_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
o
gradients/Pow_grad/mulMulgradients/Sum_grad/TilePow/y*'
_output_shapes
:џџџџџџџџџ*
T0
]
gradients/Pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
gradients/Pow_grad/subSubPow/ygradients/Pow_grad/sub/y*
_output_shapes
: *
T0
l
gradients/Pow_grad/PowPowsubgradients/Pow_grad/sub*'
_output_shapes
:џџџџџџџџџ*
T0

gradients/Pow_grad/mul_1Mulgradients/Pow_grad/mulgradients/Pow_grad/Pow*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
gradients/Pow_grad/SumSumgradients/Pow_grad/mul_1(gradients/Pow_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/Pow_grad/ReshapeReshapegradients/Pow_grad/Sumgradients/Pow_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
a
gradients/Pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
gradients/Pow_grad/GreaterGreatersubgradients/Pow_grad/Greater/y*'
_output_shapes
:џџџџџџџџџ*
T0
T
gradients/Pow_grad/LogLogsub*'
_output_shapes
:џџџџџџџџџ*
T0
a
gradients/Pow_grad/zeros_like	ZerosLikesub*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
gradients/Pow_grad/SelectSelectgradients/Pow_grad/Greatergradients/Pow_grad/Loggradients/Pow_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
o
gradients/Pow_grad/mul_2Mulgradients/Sum_grad/TilePow*
T0*'
_output_shapes
:џџџџџџџџџ

gradients/Pow_grad/mul_3Mulgradients/Pow_grad/mul_2gradients/Pow_grad/Select*
T0*'
_output_shapes
:џџџџџџџџџ
Ѕ
gradients/Pow_grad/Sum_1Sumgradients/Pow_grad/mul_3*gradients/Pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/Pow_grad/Reshape_1Reshapegradients/Pow_grad/Sum_1gradients/Pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/Pow_grad/tuple/group_depsNoOp^gradients/Pow_grad/Reshape^gradients/Pow_grad/Reshape_1
к
+gradients/Pow_grad/tuple/control_dependencyIdentitygradients/Pow_grad/Reshape$^gradients/Pow_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Pow_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Я
-gradients/Pow_grad/tuple/control_dependency_1Identitygradients/Pow_grad/Reshape_1$^gradients/Pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Pow_grad/Reshape_1*
_output_shapes
: 
b
gradients/sub_grad/ShapeShape
prediction*
_output_shapes
:*
T0*
out_type0
l
gradients/sub_grad/Shape_1Shapesecond_placeholder*
_output_shapes
:*
T0*
out_type0
Д
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Д
gradients/sub_grad/SumSum+gradients/Pow_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
И
gradients/sub_grad/Sum_1Sum+gradients/Pow_grad/tuple/control_dependency*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
к
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
р
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
e
gradients/prediction_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
d
!gradients/prediction_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Щ
/gradients/prediction_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/prediction_grad/Shape!gradients/prediction_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Т
gradients/prediction_grad/SumSum+gradients/sub_grad/tuple/control_dependency/gradients/prediction_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ќ
!gradients/prediction_grad/ReshapeReshapegradients/prediction_grad/Sumgradients/prediction_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ц
gradients/prediction_grad/Sum_1Sum+gradients/sub_grad/tuple/control_dependency1gradients/prediction_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ё
#gradients/prediction_grad/Reshape_1Reshapegradients/prediction_grad/Sum_1!gradients/prediction_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
*gradients/prediction_grad/tuple/group_depsNoOp"^gradients/prediction_grad/Reshape$^gradients/prediction_grad/Reshape_1
і
2gradients/prediction_grad/tuple/control_dependencyIdentity!gradients/prediction_grad/Reshape+^gradients/prediction_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/prediction_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
ы
4gradients/prediction_grad/tuple/control_dependency_1Identity#gradients/prediction_grad/Reshape_1+^gradients/prediction_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients/prediction_grad/Reshape_1*
_output_shapes
: 
И
gradients/MatMul_grad/MatMulMatMul2gradients/prediction_grad/tuple/control_dependencyCast*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
О
gradients/MatMul_grad/MatMul_1MatMulfirst_placeholder2gradients/prediction_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ф
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџ
с
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1

gradients/Cast_grad/CastCast0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*

DstT0*

SrcT0
b
GradientDescent/learning_rateConst*
valueB
 *ЌХ'7*
dtype0*
_output_shapes
: 

"GradientDescent/update_weight/CastCastGradientDescent/learning_rate*
_output_shapes
: *

DstT0*

SrcT0*
_class
loc:@weight
ч
2GradientDescent/update_weight/ApplyGradientDescentApplyGradientDescentweight"GradientDescent/update_weight/Castgradients/Cast_grad/Cast*
_output_shapes

:*
use_locking( *
T0*
_class
loc:@weight
№
0GradientDescent/update_bias/ApplyGradientDescentApplyGradientDescentbiasGradientDescent/learning_rate4gradients/prediction_grad/tuple/control_dependency_1*
_output_shapes
: *
use_locking( *
T0*
_class
	loc:@bias

GradientDescentNoOp1^GradientDescent/update_bias/ApplyGradientDescent3^GradientDescent/update_weight/ApplyGradientDescent
*
initNoOp^bias/Assign^weight/Assign

init_all_tablesNoOp
(
legacy_init_opNoOp^init_all_tables
P

save/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_625fdd65a3fb4ff39616b62a71a5386e/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
|
save/SaveV2/tensor_namesConst"/device:CPU:0*!
valueBBbiasBweight*
dtype0*
_output_shapes
:
v
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbiasweight"/device:CPU:0*
dtypes
2
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0

save/RestoreV2/tensor_namesConst"/device:CPU:0*!
valueBBbiasBweight*
dtype0*
_output_shapes
:
y
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B *
dtype0*
_output_shapes
:
Є
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes

::*
dtypes
2

save/AssignAssignbiassave/RestoreV2*
use_locking(*
T0*
_class
	loc:@bias*
validate_shape(*
_output_shapes
: 

save/Assign_1Assignweightsave/RestoreV2:1*
_output_shapes

:*
use_locking(*
T0*
_class
loc:@weight*
validate_shape(
8
save/restore_shardNoOp^save/Assign^save/Assign_1
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"
trainable_variables
B
weight:0weight/Assignweight/read:02weight/initial_value:08
:
bias:0bias/Assignbias/read:02bias/initial_value:08"
train_op

GradientDescent"
	variables
B
weight:0weight/Assignweight/read:02weight/initial_value:08
:
bias:0bias/Assignbias/read:02bias/initial_value:08"$
legacy_init_op

legacy_init_op*
predict_valuex
9
input_value*
first_placeholder:0џџџџџџџџџ
output_value
	truediv:0 tensorflow/serving/predict*
serving_default
4
inputs*
first_placeholder:0џџџџџџџџџ.
outputs#
prediction:0џџџџџџџџџtensorflow/serving/regress