# Lazy Rearrangement Planning in Confined Spaces

### Project Overview
This research project tackles a challenging object rearrangement problem where top-down grasps are disallowed in confined workspaces such as shelves or fridges. Therefore, the problem is much harder than the tabletop one where top-down grasps simplify robot-object interactions and waive object-object collisions.
Therefore, finding the right sequence of pick-and-place actions with which the objects are rearranged is critical to successfully fulfill the task (avoid collisions, fast computation time and fewer buffers/additional actions to use). Finding such a task sequence require expensive computational resource from both motion planning (compute a single pick-and-place action without incurring undesirable collisions and) and task planning (search the right sequence of such pick-and-place actions). This project, built on top of the [previous ICRA work](https://arxiv.org/abs/2110.02814), introduces a lazy evaluation framework to tame the combinatorial challenges of confined spaces rearrangement. It achieves significant speed-ups of computing a solution and can scale up to 16 object, outperforming other state-of-art methods in this domain.

Below you can find a [Pybullet](https://pybullet.org/wordpress/) simulation example of a robot (Motoman SDA10F) being tasked to rearrange cylindrical objects in a cluttered and confined space (a cubic workspace with transparent glasses) with the solution computed by the proposed method. The rearrangment goals can be user-defined and here the goal is to rearrange all the objects so that the objects with the same color are aligned in the same column (similar to a grocery scenario where commercial products of the same category are rearranged to be aligned after customers randomly drop them somewhere).

<img src="image_materials/sim_example.gif" />


**Real robot demo videos:**

**Paper:** (The paper link will be provided very soon)

**Citation:**
If you find this repository helpful to your research, I will be very happy and I gently request you to cite the paper:

```
@article{wang2022lazy,
  title={Lazy Rearrangement Planning in Confined Spaces},
  author={Rui Wang, Kai Gao, Jingjin Yu, Kostas E. Bekris},
  booktitle={Proceedings of the International Conference on Automated Planning and Scheduling},
  year={2022}
}
```

### Methods and Structures
This project implements 7 methods, each of which is described below
* LazyCIRSMIX3: our proposed method LRS_hybrid in this project (**the best and is recommended**)
* LazyCIRSMIX2: baseline of our proposed method, conservative version (LRS_conservative)
* LazyCIRSMIX: baseline of our proposed method, greedy version (LRS_greedy)
* CIRSMIX: improved version of the work in [this paper](https://arxiv.org/abs/2110.02814) 
* CIRS: state-of-art method from [this paper](https://arxiv.org/abs/2110.02814)
* DFSDP: state-of-art method from [this paper](https://arxiv.org/abs/2101.12241)
* mRS: state-of-art method from [this paper](https://www.researchgate.net/profile/James-Kuffner/publication/221071580_Manipulation_Planning_Among_Movable_Obstacles/links/00b7d519c6f398d2ad000000/Manipulation-Planning-Among-Movable-Obstacles.pdf)

The implementation structure is demonstrated in the following diagram. On the high level, it follows general Task and Motion Planning (TAMP) hierarchy to combine a task planner (yellow module) and a motion planner (blue module). In the structure of task planner, a local monotone solver (green module) is first designed and then integrated into a global planner (orange module) as primitives to tackle more general, non-monotone instances. Here "RearrangementTaskPlanner" (light orange module) is an abstract class for the global planner and is inherited by actual task planners (LazyCIRSMIX3, CIRS, DFSDP, mRS, etc., dark orange modules) with the naming format "Unidir\<method name\>Planner" while "MonotoneLocalSolver" (light green module) is an abstract class for the local monotone solver and is inherited by actual local solvers (LazyCIRSMIX3, CIRS, DFSDP, mRS, etc., dark green modules) with the naming format "\<method name\>Solver". If you want to use the same enviroment and compare with the methods in this work, you can follow this structure to integrate your task planners with the minimum efforts of changing the codes.

### Getting Started
The software infrastructure is developed in Ubuntu 20.04 with ROS Noetic and Python 3.8.10.
The robot physics simulator is Pybullet 3.1.3. Pybullet can be downloaded [here](https://pypi.org/project/pybullet/).
Below are the addtional dependencies you need to install so as to run the code smoothly.
matplotlib <br/>
Numpy>=1.11.0 <br/>
OpenCV 4.2.0 <br/>
Scipy <br/>
IPython <br/>
Pickle <br/>
It could be difficult to exhaustively list all potential dependencies. Should you have any difficulties trying the software, do not hesitate to contact wrui1223@gmail.com for further help.

### Instruction
Once you download the repository, you need to create a ROS workspace where this repository fits in as a ROS package. Instructions on how to create a ROS workspace can be found [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace). 
After you create a workspace, say you name the workspace as `catkin_ws`, go to the directory of the workspace and build code in the catkin workspace by running <br/>
`catkin_make` <br/>
It may throw out some minor errors and if this is the case, repeat the `catkin_make` two or three more times should work. (At least it works in my case. Again, feel free to contact wrui1223@gmail.com for further help.) <br/>
Once the `catkin_make` is successful, do not forget to do `source devel/setup.bash` in the workspace. <br/>

To try an example on any existing method, run the following <br/>
`roslaunch uniform_object_rearrangement run_example.launch run_example:="<#object> <instance_id> <generate/load an instance> <time_allowed> <method_name>"` <br/>
Here the placedholders in <> are </br>
- **<#object>**: the number of object you want to try (options 6-12)
- **<instance_id>**: which instance do you want to try (an integer)
- **<generate/load an instance>**: 'g': indicates generating a new instance; 'l': indicates loading an existing instance
- **<time_allowed>**: the time allowed for the method to solve the instance/problem (time suggestion: 120 or 240 seconds)
- **<method_name>**: indicates the name of the method you want to try (methods are provided in the section of Methods and Structures)

