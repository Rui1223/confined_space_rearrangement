/* This cpp file defines a graph which is constructed 
from the roadmap built in robotic scenarios.*/

#include <vector>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <typeinfo>
#include <cmath>

#include "Graph.hpp"

// Graph_t::Graph_t(std::string samples_file, std::string connections_file)
// {
//     // m_nNodes = nsamples;
//     specify_nodeStates(samples_file);
//     specify_neighborCosts(connections_file);
//     specify_edgeStatus();
//     // printStates();
//     // printNeighbors();
//     // printEdgeCosts();
// }

void Graph_t::constructGraph(std::string samples_file, std::string connections_file, bool isLabeled)
{
    // m_nNodes = nsamples;
    specify_nodeStates(samples_file);
    if (isLabeled == true) {
        specify_neighborCostsAndLabels(connections_file);
    }
    else {
        specify_neighborCosts(connections_file);
    }
    specify_edgeStatus();
    // printStates();
    // printNeighbors();
    // printEdgeCosts();    
}


float Graph_t::computeDist(std::vector<float> n1, std::vector<float> n2) {
    float temp_dist = 0.0;
    for (int j=0; j < n1.size(); j++) {
        temp_dist += pow(n1[j] - n2[j], 2);
    }
    temp_dist = sqrt(temp_dist);
    return temp_dist;
}


void Graph_t::specify_edgeStatus()
{
    int iter = 0;
    while (iter != m_nNodes) {
        m_edgeStatus.push_back(std::vector<int>(m_nNodes, 0));
        iter++;
    }
}

void Graph_t::resetEdgeStatus()
{
    for (int row_idx=0; row_idx<m_edgeStatus.size(); row_idx++) {
        m_edgeStatus[row_idx] = std::vector<int>(m_nNodes, 0);
    }
}


void Graph_t::specify_neighborCosts(std::string connections_file)
{
    int iter = 0;
    // create empty neighbors and cost vector for each node
    while (iter != m_nNodes) {
        m_nodeNeighbors.push_back(std::vector<int>());
        m_edgeCosts.push_back(std::vector<float>(m_nNodes, std::numeric_limits<float>::max()));
        iter++;
    }
    // read in the connection file
    m_inFile_.open(connections_file);
    // Check that the file was opened successfully
    if (!m_inFile_)
    {
        std::cerr << "Unable to open the connections file\n";
        exit(1); // call system to stop
    }
    std::string temp_str;
    int temp_n1;
    int temp_n2;
    float temp_cost;
    while (std::getline(m_inFile_, temp_str))
    {
        std::stringstream ss(temp_str);
        ss >> temp_n1 >> temp_n2 >> temp_cost;
        m_nodeNeighbors[temp_n1].push_back(temp_n2);
        m_nodeNeighbors[temp_n2].push_back(temp_n1);
        m_edgeCosts[temp_n1][temp_n2] = temp_cost;
        m_edgeCosts[temp_n2][temp_n1] = temp_cost;
    }
    m_inFile_.close();
}

void Graph_t::specify_nodeStates(std::string samples_file)
{
    // read in the samples file
    m_inFile_.open(samples_file);
    // Check that the file was opened successfully
    if (!m_inFile_)
    {
        std::cerr << "Unable to open the samples file\n";
        exit(1); // call system to stop
    }
    std::string temp_str;
    int temp_nodeIdx;
    float c;
    /// read through samples file to construct m_nodeStates
    while (std::getline(m_inFile_, temp_str))
    {
        std::stringstream ss(temp_str);
        ss >> temp_nodeIdx;
        std::vector<float> temp_v;
        while (ss >> c)
        {
            temp_v.push_back(c);
        }
        m_nodeStates.push_back(temp_v);
    }
    m_inFile_.close();

    m_nNodes = m_nodeStates.size();
}

void Graph_t::specify_neighborCostsAndLabels(std::string connections_file)
{
    int iter = 0;
    // initialize m_nodeNeighbors, m_edgeCosts, 
    // m_edgeLabels_arm, m_edgeInHandValidity, m_edgeLabels_objectInHand
    while (iter != m_nNodes) {
        m_nodeNeighbors.push_back(std::vector<int>());
        m_edgeCosts.push_back(std::vector<float>(m_nNodes, std::numeric_limits<float>::max()));
        m_edgeLabels_arm.push_back(std::vector<std::vector<int>>(m_nNodes, std::vector<int>()));
        m_edgeInHandValidity.push_back(std::vector<bool>(m_nNodes, true));
        m_edgeLabels_objectInHand.push_back(std::vector<std::vector<int>>(m_nNodes, std::vector<int>()));
        iter++;
    }
    // read in the connection file
    m_inFile_.open(connections_file);
    // Check that the file was opened successfully
    if (!m_inFile_)
    {
        std::cerr << "Unable to open the connections file\n";
        exit(1); // call system to stop
    }
    std::string temp_str;
    float c;
    int temp_n1;
    int temp_n2;
    float temp_cost;
    while (std::getline(m_inFile_, temp_str))
    {
        std::stringstream ss(temp_str);
        // first read in the two node idx and the cost
        ss >> temp_n1 >> temp_n2 >> temp_cost;
        m_nodeNeighbors[temp_n1].push_back(temp_n2);
        m_nodeNeighbors[temp_n2].push_back(temp_n1);
        m_edgeCosts[temp_n1][temp_n2] = temp_cost;
        m_edgeCosts[temp_n2][temp_n1] = temp_cost;
        bool readSecondPartLabels = false;
        while (ss >> c) {
            // check if the value is negative
            if (c < 0) {
                // this is for m_edgeInHandValidity
                if (c == -2) {
                    m_edgeInHandValidity[temp_n1][temp_n2] = false;
                    m_edgeInHandValidity[temp_n2][temp_n1] = false;
                }
                readSecondPartLabels = true;
            }
            if (c >= 0 && readSecondPartLabels) {
                // this is for m_edgeLabels_objectInHand
                m_edgeLabels_objectInHand[temp_n1][temp_n2].push_back(c);
                m_edgeLabels_objectInHand[temp_n2][temp_n1].push_back(c);
            }
            if (c >= 0 && !readSecondPartLabels) {
                // this is for m_edgeLabels_arm
                m_edgeLabels_arm[temp_n1][temp_n2].push_back(c);
                m_edgeLabels_arm[temp_n2][temp_n1].push_back(c);
            }
        }
    } 
    m_inFile_.close();
}


void Graph_t::modifyEdge(std::vector<confined_space_rearrangement::Edge> &violated_edges, int query_idx)
{
    for (auto const &edge : violated_edges) {
        // std::cout << "========Now modify graph in c++============\n";
        m_edgeStatus[edge.idx1][edge.idx2] = query_idx; // disable this edge
        m_edgeStatus[edge.idx2][edge.idx1] = query_idx;
        // std::cout << "after modification";
        // std::cout << "m_edgeStatus: " << m_edgeStatus[edge.idx1][edge.idx2] << "\n";
        // std::cout << "m_edgeStatus: " << m_edgeStatus[edge.idx2][edge.idx1] << "\n";
    }
}

void Graph_t::printStates()
{
    for (auto const &state : m_nodeStates) {
        for (auto const &e : state) {
            std::cout << e << " ";
        }
        std::cout << "\n";
    }
}

void Graph_t::printNeighbors()
{
    for (auto const &neighbors : m_nodeNeighbors) {
        for (auto const &id : neighbors) {
            std::cout << id << " ";
        }
        std::cout << "\n";
    }
}

void Graph_t::printEdgeCosts()
{
    for (auto const &costs : m_edgeCosts) {
        for (auto const &value : costs) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
}

void Graph_t::printEdgeLabelsArm_specific(int id1, int id2)
{   
    std::cout << "checking the edge: " << id1 << "\t" << id2 << "\n";
    std::vector<int> temp_edgelabelsArm = getEdgeLabelsArm(id1, id2);
    for (int i = 0; i < temp_edgelabelsArm.size(); i++) {
        std::cout << temp_edgelabelsArm[i] << " ";
    }
    std::cout << "\n";
}

void Graph_t::printEdgeLabelsInHand_specific(int id1, int id2)
{   
    std::cout << "checking the edge: " << id1 << "\t" << id2 << "\n";
    std::vector<int> temp_edgelabelsInHand = getEdgeLabelsInHand(id1, id2);
    for (int i = 0; i < temp_edgelabelsInHand.size(); i++) {
        std::cout << temp_edgelabelsInHand[i] << " ";
    }
    std::cout << "\n";
}


void Graph_t::printEdgeInHandValidity_specific(int id1, int id2)
{   
    std::cout << "checking the edge: " << id1 << "\t" << id2 << "\n";
    std::cout << getEdgeInHandValidity(id1, id2);
    std::cout << "\n";
}


// void Graph_t::connectStartAndGoal(std::string task_file)
// {
//     // specify start and goal
//     m_start = m_nNodes - 2;
//     m_goal = m_nNodes - 1;
//     /// Now read in the task file to get start & goal information
//     m_inFile_.open(task_file);
//     if (!m_inFile_)
//     {
//         std::cerr << "Unable to open the task file\n";
//         exit(1); // call system to stop     
//     }

//     int nline = 0;
//     std::string temp_str;
//     while (std::getline(m_inFile_, temp_str))
//     {
//         nline += 1;
//         std::stringstream ss(temp_str);

//         if (nline == 1 or nline == 2) {
//             int temp_nodeIdx;
//             ss >> temp_nodeIdx;
//             std::vector<float> temp_d;
//             float d;
//             while (ss >> d)
//             {
//                 temp_d.push_back(d);
//             }
//             m_nodeStates.push_back(temp_d);
//         }
//         else {
//             // add connection information for the start and goal
//             int temp_n1;
//             int temp_n2;
//             float temp_cost;
//             ss >> temp_n1 >> temp_n2 >> temp_cost;
//             m_nodeNeighbors[temp_n1].push_back(temp_n2);
//             m_nodeNeighbors[temp_n2].push_back(temp_n1);
//             m_edgeCosts[temp_n1][temp_n2] = temp_cost;
//             m_edgeCosts[temp_n2][temp_n1] = temp_cost;

//         }

//     }
//     m_inFile_.close();
    
//     m_startNode = m_nodeStates[m_start];
//     m_goalNode = m_nodeStates[m_goal];

// }