#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ros/ros.h>
#include <ros/package.h>
#include <confined_space_rearrangement/AstarPathFindingLabeled.h>
#include <confined_space_rearrangement/AstarPathFindingNonLabeled.h>
#include <confined_space_rearrangement/ResetRoadmap.h>

#include "Graph.hpp"
#include "AstarSolver.hpp"
#include "Timer.hpp"

// struct AstarResult
// {
//     bool searchSuccess;
//     std::vector<int> path;
// };

class Planner_t {

public:

    Graph_t m_right_torso_g;
    Graph_t m_right_torso_normal_g;
    AstarSolver_t m_astar_solver;

    // constructor
    Planner_t() {}
    Planner_t(std::string samples_file, std::string connections_file, std::string samples_normal_file, std::string connections_normal_file) 
    {
        m_right_torso_g.constructGraph(samples_file, connections_file, true);
        m_right_torso_normal_g.constructGraph(samples_normal_file, connections_normal_file, false);
    }

    bool astarSolverNonLabeledCallback(
        confined_space_rearrangement::AstarPathFindingNonLabeled::Request &req,
        confined_space_rearrangement::AstarPathFindingNonLabeled::Response &resp) 
    {
        if (req.armType == "Right_torso"){
            if (m_astar_solver.getQueryIdx() != req.query_idx) {
                // this is a new query, let's set the new query
                m_astar_solver.setPlanningQuery_nonLabeled(m_right_torso_normal_g, req.query_idx, 
                    req.start_idx, req.goal_idx, req.start_config, req.goal_config,
                    req.start_neighbors_idx, req.goal_neighbors_idx,
                    req.start_neighbors_cost, req.goal_neighbors_cost,
                    req.violated_edges);
            }
            m_right_torso_normal_g.modifyEdge(req.violated_edges, req.query_idx);
            m_astar_solver.prepareToSearch(m_right_torso_normal_g);
            m_astar_solver.Astar_search_nonLabeled(m_right_torso_normal_g);
        }
        // let's return the response after a search
        resp.searchSuccess = m_astar_solver.getSearchSuccessInfo();
        resp.path = m_astar_solver.getPath();
        return true;
    }

    bool astarSolverLabeledCallback(
        confined_space_rearrangement::AstarPathFindingLabeled::Request &req,
        confined_space_rearrangement::AstarPathFindingLabeled::Response &resp) 
    {
        if (req.armType == "Right_torso"){
            if (m_astar_solver.getQueryIdx() != req.query_idx) {
                // this is a new query, let's set the new query
                m_astar_solver.setPlanningQuery_labeled(m_right_torso_g, req.query_idx, 
                    req.start_idx, req.goal_idx, req.start_config, req.goal_config,
                    req.start_neighbors_idx, req.goal_neighbors_idx,
                    req.start_neighbors_cost, req.goal_neighbors_cost,
                    req.occupied_labels, req.isInHandManipulation, 
                    req.violated_edges);
            }
            m_right_torso_g.modifyEdge(req.violated_edges, req.query_idx);
            m_astar_solver.prepareToSearch(m_right_torso_g);
            m_astar_solver.Astar_search_labeled(m_right_torso_g);
        }
        // let's return the response after a search
        resp.searchSuccess = m_astar_solver.getSearchSuccessInfo();
        resp.path = m_astar_solver.getPath();
        return true;
    }

    bool resetRoadmapCallback(
        confined_space_rearrangement::ResetRoadmap::Request &req,
        confined_space_rearrangement::ResetRoadmap::Response &resp)
    {
        if (req.armType == "Right_torso") {
            m_right_torso_g.resetEdgeStatus();
            m_right_torso_normal_g.resetEdgeStatus();
        }
        resp.success = true;
        return true;
    }

};

int main(int argc, char** argv)
{
    // initialize the ros node
    ros::init(argc, argv, "main_planner");
    ros::NodeHandle nh;

    ROS_INFO_STREAM("initialize main_plannner_node");

    std::string package_path = ros::package::getPath("confined_space_rearrangement");
    Timer t;

    // load the roadmap for left and right arm
    std::string right_torso_samples_file = package_path + "/roadmaps/samples_Right_torso.txt";
    std::string right_torso_connections_file = package_path + "/roadmaps/connections_Right_torso.txt";
    std::string right_torso_samples_normal_file = package_path + "/roadmaps/samples_Right_torso_normal.txt";
    std::string right_torso_connections_normal_file = package_path + "/roadmaps/connections_Right_torso_normal.txt";
    Planner_t planner(
        right_torso_samples_file, right_torso_connections_file, right_torso_samples_normal_file, right_torso_connections_normal_file);

    // planner.printWrapper();
    std::cout << "time to load graph with " << planner.m_right_torso_g.getnNodes() << " nodes for two graphs is " << t.elapsed() << "\n";

    // claim service the node provide (server)
    ros::ServiceServer astar_nonlabeled_server = nh.advertiseService("astar_path_finding_nonlabeled", &Planner_t::astarSolverNonLabeledCallback, &planner);
    ros::ServiceServer astar_labeled_server = nh.advertiseService("astar_path_finding_labeled", &Planner_t::astarSolverLabeledCallback, &planner);
    ros::ServiceServer reset_roadmap_server = nh.advertiseService("reset_roadmap", &Planner_t::resetRoadmapCallback, &planner);


    // Loop at 2Hz until the node is shut down
    // ros::Rate rate(2);
    while (ros::ok()) {
        ros::spin();
        // rate.sleep();
    }

    return 0;

}