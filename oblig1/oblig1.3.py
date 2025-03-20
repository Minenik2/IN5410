import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Configuration
num_households = 30
num_hours = 24
max_power_per_hour = 15000  # Max power limit per hour (adjustable)
peak_hours = [17, 18, 19]  # Peak hours: 5 PM - 8 PM
ev_ownership_fraction = 0.4  # 40% of households own an EV

# Non-shiftable appliances (always on during specified hours)
non_shiftable_appliances = {
    "lighting": {"power": 1.5, "start": 10, "end": 20},  
    "heating": {"power": 7.0, "start": 0, "end": 24},  
    "refrigerator": {"power": 1.32, "start": 0, "end": 24},  
    "electric_stove": {"power": 3.9, "start": 0, "end": 24},
    "tv": {"power": 0.375, "start": 17, "end": 22},  
    "computer": {"power": 0.6, "start": 15, "end": 16},  
}

# Shiftable appliances (can be scheduled flexibly)
base_shiftable_appliances = {
    "dishwasher": {"power": 1.44, "start": 0, "end": 24},
    "washing_machine": {"power": 1.94, "start": 0, "end": 24},
    "cloth_dryer": {"power": 2.5, "start": 0, "end": 24},
    "coffee_maker": {"power": 0.264, "start": 0, "end": 24},
    "ceiling_fan": {"power": 0.225, "start": 0, "end": 24},
    "hair_dryer": {"power": 0.25, "start": 0, "end": 24},
}

# Generate Real-Time Pricing (RTP) curve
def generate_rtp():
    rtp = np.zeros(num_hours)
    for hour in range(num_hours):
        if hour in peak_hours:
            rtp[hour] = random.uniform(0.8, 1.2)  # Higher price in peak hours
        else:
            rtp[hour] = random.uniform(0.4, 0.6)  # Lower price in off-peak hours
    return rtp

pricing_curve = generate_rtp()

# Generate non-shiftable appliance schedules
def generate_non_shiftable_schedule():
    schedule = np.zeros((len(non_shiftable_appliances), num_hours))
    for i, (appliance, details) in enumerate(non_shiftable_appliances.items()):
        schedule[i, details["start"]:details["end"]] = details["power"]  # Always ON during this period
    return schedule

# Optimize scheduling for shiftable appliances
def optimize_schedule(shiftable_appliances):
    appliances_list = list(shiftable_appliances.keys())
    num_appliances = len(appliances_list)

    c = []  # Cost coefficients (penalize peak-hour usage)
    A_eq = []
    b_eq = []
    A_ub = np.zeros((num_hours, num_appliances * num_hours))  # Max power constraints
    b_ub = np.full(num_hours, max_power_per_hour)  # Max power per hour

    # Build cost vector
    for appliance in appliances_list:
        power = shiftable_appliances[appliance]["power"]
        for t in range(num_hours):
            penalty = 5 if t in peak_hours else 0  # Large penalty to avoid peak hours
            c.append((power * pricing_curve[t]) + penalty)  # Cost for using this appliance at this hour

    # Build equality constraints for required energy per appliance
    for i, appliance in enumerate(appliances_list):
        power = shiftable_appliances[appliance]["power"]
        start, end = shiftable_appliances[appliance]["start"], shiftable_appliances[appliance]["end"]

        # Appliance must run for its required energy level
        constraint_row = [0] * (num_appliances * num_hours)
        for t in range(start, end):  # Only allow usage in valid hours
            constraint_row[i * num_hours + t] = 1  

        A_eq.append(constraint_row)
        b_eq.append(power)  # Required energy per appliance

    # Build inequality constraints for max power limit per hour
    for t in range(num_hours):
        for i, appliance in enumerate(appliances_list):
            power = shiftable_appliances[appliance]["power"]
            A_ub[t, i * num_hours + t] = power  # Contribution to total power at time t

    # Convert lists to NumPy arrays
    A_eq = np.array(A_eq)
    c = np.array(c)
    b_eq = np.array(b_eq)

    # Solve using linear programming
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * len(c), method='highs')

    return result, appliances_list

# Run optimization for all households
all_non_shiftable_schedules = []
all_shiftable_schedules = []
all_shiftable_appliance_lists = []

for household in range(num_households):
    # Generate non-shiftable schedule
    non_shiftable_schedule = generate_non_shiftable_schedule()  

    # Determine if household owns an EV
    shiftable_appliances = base_shiftable_appliances.copy()
    if random.random() < ev_ownership_fraction:
        shiftable_appliances[f"ev_{household}"] = {"power": 9.9, "start": 0, "end": 24}  # Add EV for this household

    # Optimize shiftable schedule
    result, shiftable_appliance_list = optimize_schedule(shiftable_appliances)

    if result.success:
        shiftable_schedule = np.array(result.x).reshape(len(shiftable_appliance_list), num_hours)
        all_shiftable_schedules.append(shiftable_schedule)
    else:
        all_shiftable_schedules.append(np.zeros((len(shiftable_appliance_list), num_hours)))  # No optimization

    all_non_shiftable_schedules.append(non_shiftable_schedule)
    all_shiftable_appliance_lists.append(shiftable_appliance_list)

# Aggregate total power usage
total_non_shiftable = np.sum(np.array(all_non_shiftable_schedules), axis=0)
# Find the max number of shiftable appliances across all households
max_shiftable_appliances = max(len(schedule) for schedule in all_shiftable_schedules)

# Pad shiftable schedules to make them uniform
padded_shiftable_schedules = []
for schedule in all_shiftable_schedules:
    num_appliances = schedule.shape[0]
    if num_appliances < max_shiftable_appliances:
        # Pad with zero rows to match the max size
        padding = np.zeros((max_shiftable_appliances - num_appliances, num_hours))
        padded_schedule = np.vstack((schedule, padding))
    else:
        padded_schedule = schedule

    padded_shiftable_schedules.append(padded_schedule)

# Convert to NumPy array
total_shiftable = np.sum(np.array(padded_shiftable_schedules), axis=0)
total_energy_usage = np.sum(total_non_shiftable, axis=0) + np.sum(total_shiftable, axis=0)

# Plot results
def plot_neighborhood_schedule():
    plt.figure(figsize=(12, 6))

    # Non-shiftable total demand
    plt.plot(range(num_hours), np.sum(total_non_shiftable, axis=0), label="Total Non-Shiftable Demand", linestyle="dotted")

    # Shiftable total demand
    plt.plot(range(num_hours), np.sum(total_shiftable, axis=0), label="Total Shiftable Demand", linestyle="solid")

    # Overlay RTP curve
    plt.plot(range(num_hours), pricing_curve * 10, label="RTP Price (scaled)", color="black", linestyle="dashdot", alpha=0.7)
    
    # Total energy demand curve
    plt.plot(range(num_hours), total_energy_usage, label="Total Energy Consumption", color="red", linewidth=2.5, alpha=0.8)

    plt.xlabel("Hour of the Day")
    plt.ylabel("Power Usage (kWh)")
    plt.title("Total Neighborhood Energy Consumption vs. RTP Pricing")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()

# Run the plot function
plot_neighborhood_schedule()
