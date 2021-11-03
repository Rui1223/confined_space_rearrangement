/* This hpp file declares a graph is constructed
from the roadmap built in robotic scenarios. */

#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <confined_space_rearrangement/Edge.h>

class Graph_t
{
    // the size of the graph
    int m_nNodes;

    // specify neighbors(edges) and edge cost of the graph
    std::vector<std::vector<int>> m_nodeNeighbors;
    std::vector<std::vector<float>> m_edgeCosts;
    std::vector<std::vector<float>> m_nodeStates;
    std::vector<std::vector<int>> m_edgeStatus;

    // labeled graph version (additional member)
    std::vector<std::vector<std::vector<int>>> m_edgeLabels_arm;
    std::vector<std::vector<bool>> m_edgeInHandValidity;
    std::vector<std::vector<std::vector<int>>> m_edgeLabels_objectInHand;

    // // specify start and goal
    // int m_start;
    // std::vector<float> m_startNode;
    // int m_goal;
    // std::vector<float> m_goalNode;

    // file reader
    std::ifstream m_inFile_;


public:
    // constructor
    Graph_t() {}
    // Graph_t(std::string samples_file, std::string connections_file);
    void constructGraph(std::string samples_file, std::string connections_file, bool isLabeled);
    void specify_nodeStates(std::string samples_file);
    void specify_neighborCosts(std::string connections_file);
    void specify_neighborCostsAndLabels(std::string connections_file);
    void specify_edgeStatus();
    // void connectStartAndGoal(std::string task_file);
    float computeDist(std::vector<float> n1, std::vector<float> n2);

    // getter
    int getnNodes() { return m_nNodes; }
    std::vector<float> getState(int idx) { return m_nodeStates[idx]; }
    // int getStart() { return m_start; }
    // int getGoal() { return m_goal; }
    // std::vector<float> getStartState() { return m_startNode; }
    // std::vector<float> getGoalState() { return m_goalNode; }
    std::vector<int> getNodeNeighbors(int id) { return m_nodeNeighbors[id]; }
    float getEdgeCost(int id1, int id2) { return m_edgeCosts[id1][id2]; }
    int getEdgeStatus(int id1, int id2) { return m_edgeStatus[id1][id2]; }
    std::vector<int> getEdgeLabelsArm(int id1, int id2) { return m_edgeLabels_arm[id1][id2]; }
    std::vector<int> getEdgeLabelsInHand(int id1, int id2) { return m_edgeLabels_objectInHand[id1][id2]; }
    bool getEdgeInHandValidity(int id1, int id2) { return m_edgeInHandValidity[id1][id2]; }
    void modifyEdge(std::vector<confined_space_rearrangement::Edge> &violated_edges, int query_idx);
    void resetEdgeStatus();

    // printer
    void printStates();
    void printNeighbors();
    void printEdgeCosts();
    void printEdgeLabelsArm_specific(int id1, int id2);
    void printEdgeLabelsInHand_specific(int id1, int id2);
    void printEdgeInHandValidity_specific(int id1, int id2);

    // destructor
    ~Graph_t() {}

};



#endif