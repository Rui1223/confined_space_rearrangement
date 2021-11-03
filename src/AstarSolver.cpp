/* This cpp file defines A* search on a given graph
with specified start and goal */

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "Graph.hpp"
#include "AstarSolver.hpp"

// AstarSolver_t::AstarSolver_t(Graph_t &g, int start, int goal)
// {
//     // initialize the start & goal
//     m_start = start;
//     m_goal = goal;
//     // essential elements for Astar search
//     computeH(g); // heuristics
//     m_G = std::vector<float>(g.getnNodes(), std::numeric_limits<float>::max());
//     m_G[m_start] = 0.0;
//     m_open.push(new AstarNode_t(m_start, m_G[m_start], nullptr));
//     m_expanded = std::vector<bool>(g.getnNodes(), false);

//     m_isFailure = false;
    
// }


void AstarSolver_t::setPlanningQuery_nonLabeled(Graph_t &g, int query_idx, 
        int start_idx, int goal_idx, std::vector<float> start_config, std::vector<float> goal_config,
        std::vector<int> start_neighbors_idx, std::vector<int> goal_neighbors_idx,
        std::vector<float> start_neighbors_cost, std::vector<float> goal_neighbors_cost,
        std::vector<confined_space_rearrangement::Edge> violated_edges
        )
{
    // get the query info
    m_query_idx = query_idx;
    m_start = start_idx;
    m_goal = goal_idx;
    m_startState = start_config;
    m_goalState = goal_config;
    m_start_neighbors_idx = start_neighbors_idx;
    m_goal_neighbors_idx = goal_neighbors_idx;
    m_start_neighbors_cost = start_neighbors_cost;
    m_goal_neighbors_cost = goal_neighbors_cost;

    // given the goal, compute the heuristics
    // m_H = std::vector<float>(g.getnNodes()+2, 0.0);
    // computeH(g); // heuristics
}

void AstarSolver_t::setPlanningQuery_labeled(Graph_t &g, int query_idx, 
        int start_idx, int goal_idx, std::vector<float> start_config, std::vector<float> goal_config,
        std::vector<int> start_neighbors_idx, std::vector<int> goal_neighbors_idx,
        std::vector<float> start_neighbors_cost, std::vector<float> goal_neighbors_cost,
        std::vector<int> occupied_labels, bool isInHandManipulation, 
        std::vector<confined_space_rearrangement::Edge> violated_edges
        )
{
    // get the query info
    m_query_idx = query_idx;
    m_start = start_idx;
    m_goal = goal_idx;
    m_startState = start_config;
    m_goalState = goal_config;
    m_start_neighbors_idx = start_neighbors_idx;
    m_goal_neighbors_idx = goal_neighbors_idx;
    m_start_neighbors_cost = start_neighbors_cost;
    m_goal_neighbors_cost = goal_neighbors_cost;

    // related to labels
    m_occupied_labels = occupied_labels;
    m_isInHandManipulation = isInHandManipulation;

    // given the goal, compute the heuristics
    // m_H = std::vector<float>(g.getnNodes()+2, 0.0);
    // computeH(g); // heuristics
}



void AstarSolver_t::prepareToSearch(Graph_t &g)
{
    m_G = std::vector<float>(g.getnNodes()+2, std::numeric_limits<float>::max());
    m_expanded = std::vector<bool>(g.getnNodes()+2, false);
    m_isSearchSuccess = true;

    clearOpenAndCLosedList();
    m_path = std::vector<int>();
    
    m_closed = std::vector<AstarNode_t*>();
    // std::cout << "is path empty? " << (m_path.empty()) << "\n";
    // std::cout << "is open list empty? " << (m_open.empty()) << "\n";
    // std::cout << "is closed list empty? " << (m_closed.empty()) << "\n";

    // put the start in the open list
    m_G[m_start] = 0.0;
    m_open.push(new AstarNode_t(m_start, m_G[m_start], nullptr));
    AstarNode_t *current = m_open.top();
    m_open.pop();
    // put start into the closed list
    m_closed.push_back(current);
    m_expanded[g.getnNodes()] = true;
    // put the start_neighbors into the open list
    for (int i = 0; i < m_start_neighbors_idx.size(); i++) {
        int neighbor_idx = m_start_neighbors_idx[i];
        m_G[neighbor_idx] = m_start_neighbors_cost[i];
        m_open.push(new AstarNode_t(neighbor_idx, m_G[neighbor_idx], current));
    }
}


AstarSolver_t::~AstarSolver_t()
{
    while (!m_open.empty())
    {
        AstarNode_t* a1 = m_open.top();
        delete a1;
        m_open.pop();
    }
    for (auto &e : m_closed) { delete e; }
}


void AstarSolver_t::clearOpenAndCLosedList()
{
    while (!m_open.empty())
    {
        AstarNode_t* a1 = m_open.top();
        delete a1;
        m_open.pop();
    }
    for (auto &e : m_closed) { delete e; }
}


void AstarSolver_t::computeH(Graph_t &g)
{
    // loop through all nodes
    for (int i=0; i < g.getnNodes(); i++) {
        m_H[i] = computeDist(g.getState(i), m_goalState);
        // m_H.push_back(computeDist(g.getState(i), g.getGoalState()));
    }
    // also compute the heuristic for the start
    m_H[m_start] = computeDist(m_startState, m_goalState);

}

float AstarSolver_t::computeDist(std::vector<float> state1, std::vector<float> state2)
{
    float temp_dist = 0.0;
    for (int k=0; k < state1.size(); k++) {
        temp_dist += pow(state1[k]-state2[k], 2);
    }
    temp_dist = sqrt(temp_dist);
    return temp_dist;
}


void AstarSolver_t::Astar_search_nonLabeled(Graph_t &g)
{
    while (!m_open.empty()){
        AstarNode_t *current = m_open.top();
        m_open.pop();
        // Now check if the current node has been expanded
        if (m_expanded[current->m_id] == true) {
            // This node has been expanded with the lowest f value for its id
            // No need to put it into the closed list
            delete current;
            continue;
        }
        // std::cout << current->m_id << "\n";
        m_closed.push_back(current);
        m_expanded[current->m_id] = true;

        if (current->m_id == m_goal) {
            // the goal is found
            std::cout << "PATH FOUND\n";
            m_pathCost = current->m_g;
            back_track_path(); // construct your path
            // pathToTrajectory(g); // get the trajectory (a sequence of configurations)

            return;
        }
        // get neighbors of the current node
        std::vector<int> neighbors = g.getNodeNeighbors(current->m_id);
        for (auto const &neighbor : neighbors)
        {   
            // check if the edge is still valid or not
            if ( g.getEdgeStatus(current->m_id, neighbor) == m_query_idx ) {continue;}
            // check if the neighbor node has been visited or extended before
            if ( m_expanded[neighbor] ) {continue;}
            if ( m_G[neighbor] > m_G[current->m_id] + g.getEdgeCost(current->m_id, neighbor) )
            {
                m_G[neighbor] = m_G[current->m_id] + g.getEdgeCost(current->m_id, neighbor);
                m_open.push(new AstarNode_t(neighbor, m_G[neighbor], current));
            }
        }
        // Consider one more case here! The goal may also be the neighbor
        auto it = std::find(m_goal_neighbors_idx.begin(), m_goal_neighbors_idx.end(), current->m_id);
        if (it != m_goal_neighbors_idx.end()) {
            // the goal is a neighbor of the current node
            int index = it - m_goal_neighbors_idx.begin();
            if ( m_G[m_goal] > m_G[current->m_id] + m_goal_neighbors_cost[index] ) {
                m_G[m_goal] = m_G[current->m_id] + m_goal_neighbors_cost[index];
                m_open.push(new AstarNode_t(m_goal, m_G[m_goal], current));
            }
        }
    }
    // You are reaching here since the open list is empty and the goal is not found
    std::cout << "The problem is not solvable. Search failed...\n\n";
    m_isSearchSuccess = false;
    return;
}

void AstarSolver_t::Astar_search_labeled(Graph_t &g)
{
    while (!m_open.empty()){
        AstarNode_t *current = m_open.top();
        m_open.pop();
        // Now check if the current node has been expanded
        if (m_expanded[current->m_id] == true) {
            // This node has been expanded with the lowest f value for its id
            // No need to put it into the closed list
            delete current;
            continue;
        }
        // std::cout << current->m_id << "\n";
        m_closed.push_back(current);
        m_expanded[current->m_id] = true;

        if (current->m_id == m_goal) {
            // the goal is found
            std::cout << "PATH FOUND\n";
            m_pathCost = current->m_g;
            back_track_path(); // construct your path
            // pathToTrajectory(g); // get the trajectory (a sequence of configurations)

            return;
        }
        // get neighbors of the current node
        std::vector<int> neighbors = g.getNodeNeighbors(current->m_id);
        for (auto const &neighbor : neighbors)
        {   
            // check if the edge is still valid or not
            if ( g.getEdgeStatus(current->m_id, neighbor) == m_query_idx ) {continue;} 
            if ( checkEdgeCarryOccupiedLabels(g.getEdgeLabelsArm(current->m_id, neighbor)) == true ) { continue; }
            if ( m_isInHandManipulation ) {
                if ( g.getEdgeInHandValidity(current->m_id, neighbor) != m_isInHandManipulation ) { continue; }
                if ( checkEdgeCarryOccupiedLabels(g.getEdgeLabelsInHand(current->m_id, neighbor)) == true ) { continue; }
            }
            // check if the neighbor node has been visited or extended before
            if ( m_expanded[neighbor] ) {continue;}
            if ( m_G[neighbor] > m_G[current->m_id] + g.getEdgeCost(current->m_id, neighbor) )
            {
                m_G[neighbor] = m_G[current->m_id] + g.getEdgeCost(current->m_id, neighbor);
                m_open.push(new AstarNode_t(neighbor, m_G[neighbor], current));
            }
        }
        // Consider one more case here! The goal may also be the neighbor
        auto it = std::find(m_goal_neighbors_idx.begin(), m_goal_neighbors_idx.end(), current->m_id);
        if (it != m_goal_neighbors_idx.end()) {
            // the goal is a neighbor of the current node
            int index = it - m_goal_neighbors_idx.begin();
            if ( m_G[m_goal] > m_G[current->m_id] + m_goal_neighbors_cost[index] ) {
                m_G[m_goal] = m_G[current->m_id] + m_goal_neighbors_cost[index];
                m_open.push(new AstarNode_t(m_goal, m_G[m_goal], current));
            }
        }
    }
    // You are reaching here since the open list is empty and the goal is not found
    std::cout << "The problem is not solvable. Search failed...\n\n";
    m_isSearchSuccess = false;
    return;
}

bool AstarSolver_t::checkEdgeCarryOccupiedLabels(const std::vector<int> &edgeLabels)
{
    // this function checks if any of the elements in m_occupied_labels
    // appears in the edge labels
    bool isEdgeCarryOccupiedLabels = true;
    for (auto const &occupied_label : m_occupied_labels) {
        if (std::find(edgeLabels.begin(), edgeLabels.end(), occupied_label) != edgeLabels.end()) {
            return isEdgeCarryOccupiedLabels;
        }
    }
    // congrats, the edge does not carry any of the occupied labels
    return false;
}

void AstarSolver_t::back_track_path()
{
    // start from the goal
    AstarNode_t *current = m_closed[m_closed.size()-1];
    while (current->m_id != m_start)
    {
        // keep backtracking the path until you reach the start
        m_path.push_back(current->m_id);
        current = current->m_parent;
    }
    // finally put the start into the path
    m_path.push_back(current->m_id);

    std::reverse(m_path.begin(), m_path.end());
    // std::cout << "final path: \n";
    // for (auto const &e : m_path) {
    //     std::cout << e << " ";
    // }
    // std::cout << "\n";
    // std::cout << "path cost: " << m_pathCost << "\n";
}

void AstarSolver_t::pathToTrajectory(Graph_t &g)
{
    // start from the start
    for (int i=m_path.size()-1; i >=0; i--)
    {
        m_trajectory.push_back(g.getState(m_path[i]));

    }
    // // print the trajectory for checking purpose
    // std::cout << "The trajectory: \n";
    // for (auto const &t : m_trajectory)
    // {
    //  for (auto const &d : t)
    //  {
    //      std::cout << d << "   ";
    //  }
    //  std::cout << "\n";
    // }
}

// void AstarSolver_t::writeTrajectory(std::string trajectory_file)
// {
//     m_outFile_.open(trajectory_file);
//     if (m_outFile_.is_open())
//     {
//         // first write in failure indicator
//         m_outFile_ << m_isFailure << "\n";
//         if (m_isFailure == false) {
//             for (auto const &t : m_trajectory)
//             {
//                 for (auto const &d : t)
//                 {
//                     m_outFile_ << d << " ";
//                 }
//                 m_outFile_ << "\n";
//             }            
//         }

//     }
//     m_outFile_.close();
// }

// void AstarSolver_t::writePath(std::string path_file)
// {
//     m_outFile_.open(path_file);
//     if (m_outFile_.is_open()) {
//         // first write in failure indicator
//         m_outFile_ << m_isFailure << "\n";
//         if (m_isFailure == false) {
//             for (auto const &t : m_path) {
//                 m_outFile_ << t << " ";
//             }
//             m_outFile_ << "\n";
//         }
//     }
//     m_outFile_.close();
// }

void AstarSolver_t::printAll()
{
    print_path();
    print_cost();
}

void AstarSolver_t::print_path()
{
    // print the path for checking purpose
    std::cout << "path: \n";
    for (int i=m_path.size()-1; i >=0; i--)
    {
        std::cout << m_path[i] << " ";
    }
    // for (auto const &waypoint : m_path)
    // {
    //  std::cout << waypoint << " ";
    // }
    std::cout << "\n";
}

void AstarSolver_t::print_cost()
{
    std::cout << "cost: " << m_pathCost << "\n";
}