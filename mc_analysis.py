from header_import import *

if __name__ == "__main__":
    
    #Number of episode
    number_of_episode = 500000

    bjack = BlackJack()
    graph = Value_Policy_Graph_Plot()


#     # RUn all three MC methods
    BlackJack_Methods = BlackJack_First_Visit_MC_Prediction_Value(number_of_episode)
    Value = BlackJack_Methods.First_Visit_MC_Prediction_Value(bjack)
    graph.Plot_Graphs_Value(Value, Value, graph_title = "First Visit MC Prediction Value | Optimal Value Function", value = True)



    BlackJack_Methods = BlackJack_MC_Prediction_Value_With_Exploring(number_of_episode)
    Value, Policy_Value, Policy = BlackJack_Methods.MC_Prediction_Value_With_Exploring(bjack)
    graph.Plot_Graphs_Value(Value, Policy_Value, graph_title = "MC Prediction Value With Exploring | Optimal Value Function")
    graph.Plot_Graphs_policy(Policy, graph_title = "MC Prediction Value With Exploring | Policy")



    BlackJack_Methods = BlackJack_MC_Prediction_Value_With_Importance_Sampling(number_of_episode)
    Value, Policy = BlackJack_Methods.MC_Prediction_Value_With_Importance_Sampling(bjack)
    graph.Plot_Graphs_Value(Value, Policy, graph_title = "MC Prediction Value With Importance Sampling | Optimal Value Function", value = True)
    graph.Plot_Graphs_policy(Policy, graph_title = "MC Prediction Value With Importance Sampling | Policy") 




