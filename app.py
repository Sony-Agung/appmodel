import streamlit as st
import numpy as np
import pandas as pd

def north_west_corner(supply, demand):
    num_suppliers = len(supply)
    num_customers = len(demand)
    
    # Initialize variables
    allocations = np.zeros((num_suppliers, num_customers))
    s = 0
    d = 0
    
    # Perform allocations
    while s < num_suppliers and d < num_customers:
        allocation = min(supply[s], demand[d])
        allocations[s][d] = allocation
        
        # Update supply and demand
        supply[s] -= allocation
        demand[d] -= allocation
        
        # Move to the next supplier or customer
        if supply[s] == 0:
            s += 1
        if demand[d] == 0:
            d += 1
    
    return allocations

def least_cost_greedy(supply, demand, distances):
    num_suppliers = len(supply)
    num_customers = len(demand)
    
    # Initialize variables
    allocations = np.zeros((num_suppliers, num_customers))
    supply_remaining = np.array(supply)
    demand_remaining = np.array(demand)
    
    # Iterate until all demand is fulfilled
    while np.sum(demand_remaining) > 0:
        # Find the minimum distance and its indices
        min_distance = np.min(distances)
        min_indices = np.argwhere(distances == min_distance)
        
        # Iterate through minimum distance indices
        for idx in min_indices:
            supplier_idx, customer_idx = idx[0], idx[1]
            
            # Calculate allocation amount
            allocation = min(supply_remaining[supplier_idx], demand_remaining[customer_idx])
            
            # Update allocations
            allocations[supplier_idx][customer_idx] += allocation
            
            # Update supply and demand
            supply_remaining[supplier_idx] -= allocation
            demand_remaining[customer_idx] -= allocation
            
            # Set distance to infinity to avoid re-allocation
            distances[supplier_idx][customer_idx] = np.inf
    
    return allocations

# Streamlit app
st.title('Biaya Pengiriman dengan Metode Northwest Corner atau Least Cost')

# Input data
method = st.radio("Choose Optimization Method:", ("North West Corner", "Least Cost"))

if method == "North West Corner":
    st.header('North West Corner Method')
    supply = []
    demand = []
    
    num_suppliers = st.number_input('Enter the number of suppliers:', min_value=1, step=1, value=3)
    num_customers = st.number_input('Enter the number of demand:', min_value=1, step=1, value=3)
    
    st.subheader('Masukan Informasi Supply:')
    gudang_names = [st.text_input(f'Enter name for Supplier {i+1}:', f'Supplier {i+1}') for i in range(num_suppliers)]
    
    for i in range(num_suppliers):
        supply_input = st.number_input(f'Enter supply for {gudang_names[i]}:', min_value=0, step=1)
        supply.append(supply_input)
        
    st.subheader('Masukan Informasi Demand:')
    toko_names = [st.text_input(f'Enter name for Customer {i+1}:', f'Customer {i+1}') for i in range(num_customers)]
    
    for j in range(num_customers):
        demand_input = st.number_input(f'Enter demand for {toko_names[j]}:', min_value=0, step=1)
        demand.append(demand_input)
    
    st.subheader('Masukan Jarak:')
    distances = np.zeros((num_suppliers, num_customers))
    for i in range(num_suppliers):
        for j in range(num_customers):
            distance_input = st.number_input(f'Distance from {gudang_names[i]} to {toko_names[j]}:', value=0)
            distances[i][j] = distance_input
    
    try:
        # Calculate allocations using NWC method
        allocations = north_west_corner(supply, demand)

        # Display allocations for NWC
        st.subheader('Allocation Table (North West Corner):')
        df_allocations = pd.DataFrame(allocations, columns=[f'{toko_names[i]}' for i in range(num_customers)])
        st.write(df_allocations)

        # Calculate total multiplication from the distribution table
        total_multiplication = np.sum(allocations * np.array(distances))

        # Display total multiplication
        st.subheader('Total Multiplication from Distribution Table:')
        st.write(f'Total Multiplication: {total_multiplication}')

        # Ask for cost per kilometer
        cost_per_km_input = st.number_input('Enter cost per kilometer (optional):', min_value=0)

        # Calculate total cost if cost_per_km is provided
        if cost_per_km_input:
            total_cost = total_multiplication * cost_per_km_input
            st.write('### Total Cost:')
            st.write(f'Total Cost: {total_cost}')

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif method == "Least Cost":
    st.header('Least Cost Method')
    supply = []
    demand = []
    
    num_suppliers = st.number_input('Enter the number of suppliers:', min_value=1, step=1, value=3)
    num_customers = st.number_input('Enter the number of customers:', min_value=1, step=1, value=3)
    
    st.subheader('Enter Supplier Information:')
    gudang_names = [st.text_input(f'Enter name for Supplier {i+1}:', f'Supplier {i+1}') for i in range(num_suppliers)]
    
    for i in range(num_suppliers):
        supply_input = st.number_input(f'Enter supply for {gudang_names[i]}:', min_value=0, step=1)
        supply.append(supply_input)
        
    st.subheader('Enter Customer Information:')
    toko_names = [st.text_input(f'Enter name for Customer {i+1}:', f'Customer {i+1}') for i in range(num_customers)]
    
    for j in range(num_customers):
        demand_input = st.number_input(f'Enter demand for {toko_names[j]}:', min_value=0, step=1)
        demand.append(demand_input)
    
    st.subheader('Enter Distances:')
    distances = np.zeros((num_suppliers, num_customers))
    for i in range(num_suppliers):
        for j in range(num_customers):
            distance_input = st.number_input(f'Distance from {gudang_names[i]} to {toko_names[j]}:', value=0)
            distances[i][j] = distance_input
    
    try:
        # Calculate allocations using Least Cost method
        allocations = least_cost_greedy(supply, demand, np.array(distances))

        # Display allocations for Least Cost
        st.subheader('Allocation Table (Least Cost):')
        df_allocations = pd.DataFrame(allocations, columns=[f'{toko_names[i]}' for i in range(num_customers)])
        st.write(df_allocations)

        # Calculate total multiplication from the distribution table
        total_multiplication = np.sum(allocations * np.array(distances))

        # Display total multiplication
        st.subheader('Total Multiplication from Distribution Table:')
        st.write(f'Total Multiplication: {total_multiplication}')

        # Ask for cost per kilometer
        cost_per_km_input = st.number_input('Enter cost per kilometer (optional):', min_value=0)

        # Calculate total cost if cost_per_km is provided
        if cost_per_km_input:
            total_cost = total_multiplication * cost_per_km_input
            st.write('### Total Cost:')
            st.write(f'Total Cost: {total_cost}')

    except Exception as e:
        st.error(f"An error occurred: {e}")
