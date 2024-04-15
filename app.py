import streamlit as st
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def northwest_corner_method(supply, demand, distances, cost_per_km):
    num_suppliers = len(supply)
    num_customers = len(demand)
    
    # Initialize variables
    allocations = np.zeros((num_suppliers, num_customers))
    supply_remaining = np.array(supply)
    demand_remaining = np.array(demand)
    
    # Northwest Corner Method
    i, j = 0, 0
    while i < num_suppliers and j < num_customers:
        # Find the minimum between supply and demand
        allocation = min(supply_remaining[i], demand_remaining[j])
        
        # Allocate the minimum amount
        allocations[i][j] = allocation
        
        # Update supply and demand
        supply_remaining[i] -= allocation
        demand_remaining[j] -= allocation
        
        # Move to the next supplier or customer
        if supply_remaining[i] == 0:
            i += 1
        else:
            j += 1
    
    # Adjust the shape of distances to match allocations
    adjusted_distances = np.zeros_like(allocations)
    adjusted_distances[:distances.shape[0], :distances.shape[1]] = distances
    
    # Calculate total cost if cost_per_km is provided
    total_cost = np.nan
    if cost_per_km:
        total_cost = np.sum(allocations * adjusted_distances) * cost_per_km
        
    return allocations, total_cost

def least_cost_parallel(supply, demand, distances, cost_per_km):
    num_suppliers = len(supply)
    num_customers = len(demand)
    
    # Initialize variables
    allocations = np.zeros((num_suppliers, num_customers))
    supply_remaining = np.array(supply)
    demand_remaining = np.array(demand)
    
    # Function to calculate least cost for a given row
    def calculate_least_cost(i):
        while np.any(supply_remaining[i]) and np.any(demand_remaining):
            # Find index with minimum cost
            min_cost_index = np.unravel_index(np.argmin(distances[i]), distances[i].shape)
            j = min_cost_index[0]
            
            # Allocate minimum between supply and demand
            allocation = min(supply_remaining[i], demand_remaining[j])
            allocations[i][j] = allocation
            
            # Update supply, demand, and distances
            supply_remaining[i] -= allocation
            demand_remaining[j] -= allocation
            distances[i][j] = np.inf
    
    # Parallelize the calculation across rows
    Parallel(n_jobs=-1)(delayed(calculate_least_cost)(i) for i in range(num_suppliers))
    
    # Calculate total cost if cost_per_km is provided
    total_cost = np.nan
    if cost_per_km:
        total_cost = np.sum(allocations * distances) * cost_per_km
        
    return allocations, total_cost

# Function to generate reference table for distances
def generate_distance_table(num_gudang, num_toko, gudang_names, toko_names):
    distances = np.zeros((num_gudang, num_toko))
    for i in range(num_gudang):
        for j in range(num_toko):
            key = f'distance_{i}_{j}'  # Generate unique key
            distance_input = st.number_input(f'Jarak dari {gudang_names[i]} ke {toko_names[j]} (km):', key=key, min_value=0, step=1)
            distances[i][j] = distance_input
    return distances

# Streamlit app
st.title('Biaya Pengiriman dengan Metode Northwest Corner atau Least Cost')

# Input data
num_gudang = st.number_input('Masukkan jumlah gudang:', min_value=1, step=1, value=3)
num_toko = st.number_input('Masukkan jumlah toko:', min_value=1, step=1, value=4)

supply = []
demand = []

st.write('### Masukkan Data:')
st.write('Isi tabel di bawah ini dengan data yang sesuai.')

# Create empty dataframe for input
input_df = pd.DataFrame(np.zeros((num_gudang + 1, num_toko)), 
                        index=[f'Gudang {i+1}' for i in range(num_gudang)] + ['Total Supply'],
                        columns=[f'Toko {i+1}' for i in range(num_toko)])
input_table = st.table(input_df)

# Input data supply
for i in range(num_gudang):
    supply_input = st.number_input(f'Masukkan persediaan di Gudang {i+1}:', min_value=0, step=1)
    supply.append(supply_input)
    input_df.iloc[i, :] = supply_input

# Input data demand
for j in range(num_toko):
    demand_input = st.number_input(f'Masukkan daya tampung di Toko {j+1}:', min_value=0, step=1)
    demand.append(demand_input)
    input_df.iloc[num_gudang, j] = demand_input

# Input data jarak
st.write('### Tabel Jarak (km):')
gudang_names = [st.text_input(f'Nama Gudang {i+1}:', f'Gudang {i+1}') for i in range(num_gudang)]
toko_names = [st.text_input(f'Nama Toko {i+1}:', f'Toko {i+1}') for i in range(num_toko)]
distances = generate_distance_table(num_gudang, num_toko, gudang_names, toko_names)
distances_df = pd.DataFrame(distances, index=gudang_names, 
                             columns=toko_names)
st.write(distances_df)

# Input biaya per kilometer (opsional)
st.write('### Asumsi Biaya per Kilometer (Rp):')
cost_per_km_input = st.number_input('Masukkan biaya per kilometer (Rp):', min_value=0)

# Select method
method = st.radio('Pilih Metode Pengalokasian:', ('Northwest Corner', 'Least Cost'))

# Add dummy row for unbalanced supply
total_supply = sum(supply)
if total_supply < sum(demand):
    supply.append(sum(demand) - total_supply)
    input_df.loc['Dummy'] = 0

# Calculate allocations
if method == 'Northwest Corner':
    allocations, total_cost = northwest_corner_method(supply, demand, distances, cost_per_km_input)
elif method == 'Least Cost':
    allocations, total_cost = least_cost_parallel(supply, demand, distances, cost_per_km_input)

# Display results
st.write('### Tabel Distribusi:')
df_allocations = pd.DataFrame(allocations, index=[f'Gudang {i+1}' for i in range(num_gudang)] + (['Dummy'] if total_supply < sum(demand) else []), 
                              columns=[f'Toko {i+1}' for i in range(num_toko)])
st.write(df_allocations)

# Calculate total multiplication from the distribution table
total_multiplication = np.sum(df_allocations.values * distances)

# Display total multiplication
st.write('### Total Hasil Perkalian dari Tabel Distribusi:')
st.write(f'Total Hasil Perkalian: {total_multiplication}')

# Display total cost if cost_per_km is provided
if cost_per_km_input:
    st.write('### Biaya Total Pengiriman:')
    if not np.isnan(total_cost):
        st.write(f'Biaya Total Pengiriman: Rp {total_cost}')
    else:
        st.write('Biaya Total Pengiriman: NaN')
