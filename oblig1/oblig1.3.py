import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Set up simulation parameters
num_households = 30
num_hours = 24
max_power_per_hour = 15000  # Max power limit per hour (adjustable)
peak_hours = [17, 18, 19]  # Peak hours: 5 PM - 8 PM
ev_ownership_fraction = 0.4  # 40% of households own an EV

# Appliance data (power consumption per hour in kWh)
non_shiftable_appliances = {
    "lighting": {"power": 1.5, "start": 10, "end": 20},  
    "heating": {"power": 7.0, "start": 0, "end": 24},  
    "refrigerator": {"power": 1.32, "start": 0, "end": 24},  
    "electric_stove": {"power": 3.9, "start": 0, "end": 24},
    "tv": {"power": 0.375, "start": 17, "end": 22},  
    "computer": {"power": 0.6, "start": 15, "end": 16},  
}

# Shiftable appliances
shiftable_appliances = {
    "dishwasher": {"power": 1.44, "start": 0, "end": 24},
    "washing_machine": {"power": 1.94, "start": 0, "end": 24},
    "cloth_dryer": {"power": 2.5, "start": 0, "end": 24},
    "ev": {"power": 9.9, "start": 0, "end": 24},
    "coffee_maker": {"power": 0.264, "start": 0, "end": 24},
    "ceiling_fan": {"power": 0.225, "start": 0, "end": 24},
    "hair_dryer": {"power": 0.25, "start": 0, "end": 24},
}

# Add EVs to a fraction of households
ev_households = int(num_households * ev_ownership_fraction)
for i in range(ev_households):
    shiftable_appliances[f"ev_{i}"] = {"power": 9.9, "start": 0, "end": 24}

# Generate random RTP pricing curve
def generate_rtp():
    rtp = np.zeros(num_hours)
    for hour in range(num_hours):
        if hour in peak_hours:
            rtp[hour] = random.uniform(0.8, 1.2)  # Peak price
        else:
            rtp[hour] = random.uniform(0.4, 0.6)  # Off-peak price
    return rtp

# Generate pricing curve for the neighborhood
pricing_curve = generate_rtp()

# Schedule Non-Shiftable Appliances for a Household
def generate_non_shiftable_schedule():
    non_shiftable_schedule = np.zeros((len(non_shiftable_appliances), num_hours))

    for i, (appliance, details) in enumerate(non_shiftable_appliances.items()):
        start, end = details["start"], details["end"]
        non_shiftable_schedule[i, start:end] = 1  # Always ON during this period
    
    return non_shiftable_schedule

# Function to optimize scheduling for appliances of a single household
def optimize_schedule():
    appliances_list = list(shiftable_appliances.keys())
    num_appliances = len(appliances_list)

    c = []  # Cost coefficients
    A_eq = []
    b_eq = []
    A_ub = np.zeros((num_hours, num_appliances * num_hours))  # Max power constraints
    b_ub = np.full(num_hours, max_power_per_hour)  # Max power per hour

    # Build cost vector, penalizing peak-hour usage
    for appliance in appliances_list:
        power = shiftable_appliances[appliance]["power"]
        for t in range(num_hours):
            penalty = 100 if t in peak_hours else 0  # High penalty to avoid peak hours
            c.append((power * pricing_curve[t]) + penalty)  # Cost for using this appliance at this hour

    # Build equality constraints for required energy per appliance
    for i, appliance in enumerate(appliances_list):
        power = shiftable_appliances[appliance]["power"]
        start, end = shiftable_appliances[appliance]["start"], shiftable_appliances[appliance]["end"]

        # Constraint: Appliance should run for its required energy level
        constraint_row = [0] * (num_appliances * num_hours)  # Full zero row
        for t in range(start, end):  # Only allow usage in valid hours
            constraint_row[i * num_hours + t] = 1  # Select variables for this appliance

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

# Function to calculate total energy cost for the entire neighborhood
def calculate_neighborhood_cost():
    total_cost = 0
    for _ in range(num_households):
        result, _ = optimize_schedule()
        if result.success:
            household_cost = result.fun  # Energy cost for the household
            total_cost += household_cost
    return total_cost

# Compute the total cost for the neighborhood
neighborhood_cost = calculate_neighborhood_cost()
print(f"Total Energy Cost for the Neighborhood: {neighborhood_cost:.2f} NOK")

# Plotting (same as before)
def plot_schedule():
    # For simplicity, we can just plot a random schedule for the first household
    non_shiftable_schedule = generate_non_shiftable_schedule()  # Generate non-shiftable schedule
    result, shiftable_appliances_list = optimize_schedule()

    if result.success:
        shiftable_schedule = np.array(result.x).reshape(len(shiftable_appliances_list), num_hours)

        # Combine schedules into one unified matrix
        total_schedule = np.vstack((non_shiftable_schedule, shiftable_schedule))
        all_appliances = list(non_shiftable_appliances.keys()) + shiftable_appliances_list

        # Compute total power consumption per hour
        total_power_usage = np.sum(total_schedule, axis=0)

        # Plot appliance usage as continuous curves
        plt.figure(figsize=(12, 6))

        # Plot non-shiftable appliances
        for i, appliance in enumerate(non_shiftable_appliances.keys()):
            plt.plot(range(num_hours), non_shiftable_schedule[i], label=f"{appliance} (fixed)", linestyle="dotted")

        # Plot shiftable appliances
        for i, appliance in enumerate(shiftable_appliances_list):
            plt.plot(range(num_hours), shiftable_schedule[i], label=f"{appliance} (optimized)")

        # Overlay the RTP curve
        plt.plot(range(num_hours), pricing_curve * 10, label="RTP Price (scaled)", color="black", linestyle="dashdot", alpha=0.7)
        
        # Plot total power consumption as a smooth curve
        plt.plot(range(num_hours), total_power_usage, label="Total Energy Consumption", color="red", linewidth=2.5, alpha=0.8)

        plt.xlabel("Hour of the Day")
        plt.ylabel("Power Usage (kWh)")
        plt.title("Optimized Appliance Scheduling vs. RTP Pricing")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.show()

# Plotting the schedule for one household as an example
plot_schedule()