import pandas as pd
import ast
import gurobipy as gp
from gurobipy import GRB, quicksum

B = 150
B_min =0
M = 10000
T_start = 540
T_end = 1380
dp_node = 'utown'

def parse_param(dp_node='DP2', node_file_path='./data/Dataset.xlsx', distance_file_path='./data/distance_in_min.xlsx'):
    '''
    parsing parameters from excel file
    node_file : node related information
    distance_file : distance between nodes

    A : Activities
    F_b : Breakfast activities
    F_l : Lunch activities
    F_d : Dinner activities
    T_set : Tour activities
    R : Attraction activities
    '''
    

    # data load
    df = pd.read_excel(node_file_path)
    df = df.fillna('')
    df = df[df.DP.isin(['',dp_node])].reset_index(drop=True)

    # place address and name mapper    
    place_mapper = dict()
    for place_name, name in df[['Place_name','Name']].values:
        place_mapper.update({name:place_name})

    # dp node
    DP = df.loc[df['DP']==dp_node,'Name'].item()

    # distance
    df_distance = pd.read_excel(distance_file_path, index_col='x')

    A = df[df.type!='start_end']['Name'].tolist()   # Activities
    F_b = df[df.type=='Fb']['Name'].tolist() # Breakfast activities
    F_l = df[df.type=='Fl']['Name'].tolist()     # Lunch activities
    F_d = df[df.type=='Fd']['Name'].tolist()    # Dinner activities
    T_set = df[df.type=='tours']['Name'].tolist()    # Tour activities
    R = df[df.type=='attractions']['Name'].tolist()  # Attraction activities

    # Nodes
    N = df['Name'].tolist()

    # Time parameters (s = earliest start time, e = latest end time)
    s = dict()
    for k, v in df[df.type!='start_end'][['Name','Start']].values:
        s.update({k:v})
        
    e = dict()
    for k, v in df[df.type!='start_end'][['Name','End']].values:
        e.update({k:v})

    # Duration parameter
    d = dict()
    for k, v in df[df.type!='start_end'][['Name','Duration']].values:
        d.update({k:v})

    # rating_Duration parameter
    # ud = dict()
    # for k, v in df[df.type!='start_end'][['Name','Rating_duration']].values:
    #     ud.update({k:v})

    # cost parameter
    c = dict()
    for k, v in df[df.type!='start_end'][['Name','Price']].values:
        c.update({k:v})

    # rating parameter
    u = dict()
    for k, v in df[df.type!='start_end'][['Name','Rating']].values:
        u.update({k:v})

    # Initialize an empty dictionary for the travel times
    t = {}

    # Iterate through each row of the DataFrame
    for index, row in df_distance.iterrows():
        t[index] = row.to_dict()

    return DP, A, F_b, F_l, F_d, T_set, R, N, e, s, d, c, u, t, place_mapper

def create_model(dp_node, B, M, T_start, T_end, B_min=0):
    DP, A, F_b, F_l, F_d, T_set, R, N, e, s, d, c, u, t, place_mapper = parse_param(dp_node=dp_node)
    '''
    create trip scheduling model
    '''
    # Create a new model
    m = gp.Model("TripScheduling")
    # Activity Selection Variables (x_i)
    x = m.addVars(A, vtype=GRB.BINARY, name="x")
    # x = m.addVars(A, vtype=GRB.CONTINUOUS, name="x")
    # Sequencing Variables (y_{ij})
    y = m.addVars(N, N, vtype=GRB.BINARY, name="y")
    # y = m.addVars(N, N, vtype=GRB.CONTINUOUS, name="y")
    # Start Time Variables (T_i)
    T = m.addVars(A, vtype=GRB.CONTINUOUS, name="T")
    # Sequence Position Variables (tau_i)
    tau = m.addVars(A, vtype=GRB.CONTINUOUS, lb=1, name="tau")

    #Trip start and ends at specific designated point (home)
    m.addConstr(gp.quicksum(y[DP, j] for j in A) == 1, name="StartAtDP") # Start of the trip
    m.addConstr(gp.quicksum(y[i, DP] for i in A) == 1, name="EndAtDP") # End of the trip

    #Trip duration from T_start to T_end
    total_duration = gp.quicksum(d[i] * x[i] for i in A) + gp.quicksum(t[i][j] * y[i, j] for i in N for j in N if i != j)
    m.addConstr(total_duration <= T_end - T_start, "TotalDuration") # inequality -- can allow some extra time (for feasibility)

    #Flow conservation for activities
    for j in A:
        # Inbound flow equals activity selection
        m.addConstr(gp.quicksum(y[i, j] for i in N if i != j) == x[j], name=f"InFlow_{j}")
        # Outbound flow equals activity selection
        m.addConstr(gp.quicksum(y[j, k] for k in N if k != j) == x[j], name=f"OutFlow_{j}")

    #Time window
    for i in A:
        m.addConstr(T[i] >= s[i], name=f"StartTimeLB_{i}")
        m.addConstr(T[i] <= e[i] - d[i], name=f"StartTimeUB_{i}")

    #Sequencing and Time
    for i in N:
        for j in N:
            if i != j:
                if i == DP:
                    # From DP to first activity
                    m.addConstr(
                        T[j] >= T_start + t[i][j] - M * (1 - y[i, j]),
                        name=f"TimeFromDP_{j}"
                    )
                elif j == DP:
                    # From last activity to DP (no T['DP'] needed)
                    pass
                else:
                    # Between activities
                    m.addConstr(
                        T[j] >= T[i] + d[i] + t[i][j] - M * (1 - y[i, j]),
                        name=f"TimeSeq_{i}_{j}"
                    )

    #Budget constraint
    m.addConstr(gp.quicksum(c[i] * x[i] for i in A) <= B, name="BudgetConstraint")
    m.addConstr(gp.quicksum(c[i] * x[i] for i in A) >= B_min, name="BudgetConstraint")
    #Food activity constraint
    if T_start >= 600:
        pass
    else:
        m.addConstr(gp.quicksum(x[i] for i in F_b) == 1, name="BreakfastConstraint")
    if T_start >= 840:
        pass
    else:
        m.addConstr(gp.quicksum(x[i] for i in F_l) == 1, name="LunchConstraint")
    m.addConstr(gp.quicksum(x[i] for i in F_d) == 1, name="DinnerConstraint")

    #Tour and attraction constraint
    m.addConstr(gp.quicksum(x[i] for i in T_set) >= 1, name="TourConstraint")
    m.addConstr(gp.quicksum(x[i] for i in R) >= 1, name="AttractionConstraint")

    #Subtour eliminatin constraint
    for i in A:
        for j in A:
            if i != j:
                m.addConstr(
                    tau[i] - tau[j] + len(A) * y[i, j] <= len(A) - 1,
                    name=f"SubtourElim_{i}_{j}"
                )

    #Domain of sequence variable
    for i in A:
        m.addConstr(tau[i] >= 1, name=f"TauLB_{i}")

    # 1st Round: Maximize Enjoyment

    # Define the objective function to maximize enjoyment
    m.setObjective(gp.quicksum(u[i] * x[i] for i in A), GRB.MAXIMIZE)

    # Optimize the model for the 1st round
    m.optimize()
    # Check if the solution is optimal
    if m.status == GRB.OPTIMAL:
        # Get the optimal objective value
        optimal_value = m.objVal
        
        # Get the best bound (this is usually the best value found during optimization)
        best_bound = m.ObjBound
        
        # Calculate the optimality gap
        optimality_gap = (best_bound - optimal_value) / best_bound * 100
        
        # Print the optimality gap
        print(f"(Enjoyment) Optimality Gap: {optimality_gap:.2f}%")
    else:
        print("Model did not find an optimal solution.")

    # Check if the optimization was successful
    if m.status == GRB.OPTIMAL:
        # Store the maximum enjoyment score from the 1st round
        max_enjoyment = m.ObjVal
        print(f"1st Round Optimization Successful: Max Enjoyment = {max_enjoyment}")
    else:
        print("1st Round Optimization Failed")
        exit()

    if m.status == GRB.INFEASIBLE:
        print("Model is infeasible. Running infeasibility analysis...")
        m.computeIIS()
        m.write("infeasible_constraints.ilp")
        
    # 2nd Round: Minimize Total Travel Time
    # Update the objective function to minimize total travel time
    m.setObjective(gp.quicksum(t[i][j] * y[i, j] for i in N for j in N if i != j), GRB.MINIMIZE)

    # Add the additional constraint for minimum enjoyment (at least 90% of the maximum enjoyment from the 1st round)
    m.addConstr(gp.quicksum(u[i] * x[i] for i in A) >= 0.90 * max_enjoyment, "MinEnjoymentConstraint")

    # Optimize the model for the 2nd round
    m.optimize()
    # Check if the solution is optimal
    if m.status == GRB.OPTIMAL:
        # Get the optimal objective value
        optimal_value = m.objVal
        
        # Get the best bound (this is usually the best value found during optimization)
        best_bound = m.ObjBound
        
        # Calculate the optimality gap
        optimality_gap = (best_bound - optimal_value) / best_bound * 100
        
        # Print the optimality gap
        print(f"(Travel time) Optimality Gap: {optimality_gap:.2f}%")
    else:
        print("Model did not find an optimal solution.")

    # Check if the 2nd round optimization was successful
    if m.status == GRB.OPTIMAL:
        total_travel_time = m.ObjVal
        total_enjoyment = gp.quicksum(u[i]*x[i].X for i in A)
        print(f"2nd Round Optimization Successful: Minimized Total Travel Time = {total_travel_time} minutes")

        # Extract the optimal solution
        selected_activities = [i for i in A if x[i].X > 0.5]
        # Get the start times of activities
        activity_start_times = {i: T[i].X for i in selected_activities}

        # Build the sequence of the trip
        sequence = []
        current_node = DP
        visited = set()
        while True:
            for j in N:
                if y[current_node, j].X > 0.5 and j != current_node and (current_node, j) not in visited:
                    sequence.append((current_node, j))
                    visited.add((current_node, j))
                    current_node = j
                    break
            else:
                break
            if current_node == DP:
                break

        # Build the itinerary
        itinerary = []
        time_pointer = T_start  # Start time of the trip
        for arc in sequence:
            i, j = arc
            if j != DP:
                # Travel from i to j
                travel_time = t[i][j]
                arrival_time = time_pointer + travel_time
                itinerary.append({
                    'start_time': time_pointer,
                    'end_time': arrival_time,
                    'description': f"Travel from {place_mapper[i]} to {place_mapper[j]}",
                    'place':i
                })

                # Activity at j
                activity_start = max(arrival_time, s[j])
                idle_time = activity_start - arrival_time
                if idle_time > 0:
                    itinerary.append({
                        'start_time': arrival_time,
                        'end_time': activity_start,
                        'description': f"Wait until {place_mapper[j]} opens"
                    })
                activity_end = activity_start + d[j]
                activity_type = (
                    'Breakfast' if j in F_b else
                    'Lunch' if j in F_l else
                    'Dinner' if j in F_d else
                    'Tour' if j in T_set else
                    'Attraction' if j in R else
                    'Activity'
                )
                itinerary.append({
                    'start_time': activity_start,
                    'end_time': activity_end,
                    'description': f"{activity_type} at {place_mapper[j]}",
                    'place':j

                })
                time_pointer = activity_end
            else:
                # Travel back to DP
                travel_time = t[i][j]
                itinerary.append({
                    'start_time': time_pointer,
                    'end_time': time_pointer + travel_time,
                    'description': "Travel back home"
                })
                time_pointer += travel_time

        # Print the itinerary
        print("\nOptimal Schedule:")
        for segment in itinerary:
            start_time = segment['start_time']
            end_time = segment['end_time']
            description = segment['description']
            start_hour = int(start_time // 60)
            start_minute = int(start_time % 60)
            end_hour = int(end_time // 60)
            end_minute = int(end_time % 60)
            start_period = 'AM' if start_hour < 12 or start_hour == 24 else 'PM'
            end_period = 'AM' if end_hour < 12 or end_hour == 24 else 'PM'
            start_hour_formatted = start_hour if start_hour <= 12 else start_hour - 12
            if start_hour_formatted == 0:
                start_hour_formatted = 12
            end_hour_formatted = end_hour if end_hour <= 12 else end_hour - 12
            if end_hour_formatted == 0:
                end_hour_formatted = 12
            print(f"## {start_hour_formatted:02d}:{start_minute:02d}{start_period} ~ {end_hour_formatted:02d}:{end_minute:02d}{end_period} : {description}")
            # print(f'u:{u[i]}')
        # Print total enjoyment and travel time
        # print(f"\nTotal Enjoyment: {total_enjoyment}")
        print(f"\nTotal Travel Time: {total_travel_time} minutes")
        total_places = []
        for i in itinerary:
            if 'place' in i.keys():
                place = i['place']
                # print(place)
                if place not in total_places:
                    total_places.append(place)
        # activity_times = 0
        # for i in itinerary:
        #     if ('Travel from' not in i['description']) & (i['description']!='Travel back home'):
        #         # print(i)
        #         duration = i['end_time']-i['start_time']
        #         activity_times+=duration
        # print(f'Total Enjoyment: {round(total_enjoyment.getValue())}/{round(activity_times/60*5)} ({round(total_enjoyment.getValue()/(activity_times/60*5), 2)})')
        print(f'Total Enjoyment: {round(total_enjoyment.getValue())}/{(len(total_places)-1)*5}')
        print(f'Total Place to visit: {(len(total_places)-1)}')
    else:
        print("2nd Round Optimization Failed")
    return itinerary, total_enjoyment

