import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import linprog

num_hours = 24
max_power_per_hour = 15  # Max power limit per hour (adjustable)
peak_hours = [17, 18, 19]  # Peak hours: 5 PM - 8 PM


# Appliance data (power consumption per hour in kWh)
# Non shiftable appliances means they have a constant stream of electricity between their start and end
non_shiftable_appliances = {
    "lighting": {"power": 1.5, "start": 10, "end": 20},  
    "heating": {"power": 7.0, "start": 0, "end": 24},  
    "refrigerator": {"power": 1.32, "start": 0, "end": 24},  
    "electric_stove": {"power": 3.9, "start": 0, "end": 24},
    "tv": {"power": 0.375, "start": 17, "end": 22},  # TV runtime is not stated start or end only that it's 5 hours therefore I made the assumption myself
    "computer": {"power": 0.6, "start": 15, "end": 16},  # assume one computer, also time is not stated neither is how many hours
}

# Appliance data (power consumption per hour in kWh)
# Shiftable appliances mean that they can choose to be turned on and off between 0 and 24
shiftable_appliances = {
    # shiftable appliances
    "dishwasher": {"power": 1.44, "start": 0, "end": 24},
    "washing_machine": {"power": 1.94, "start": 0, "end": 24},
    "cloth_dryer": {"power": 2.5, "start": 0, "end": 24},
    "ev": {"power": 9.9, "start": 0, "end": 24},
    # extra appliances
    # Power is based on the energyusecalculator.com, with the default hours used per day values
    "coffee_maker": {"power": 0.264, "start": 0, "end": 24},
    "ceiling_fan": {"power": 0.225, "start": 0, "end": 24},
    "hair_dryer": {"power": 0.25, "start": 0, "end": 24},
}

# Generate random RTP pricing curve
def generate_rtp():
    rtp = np.zeros(num_hours)
    for hour in range(num_hours):
        if hour in peak_hours:
            rtp[hour] = random.uniform(0.8, 1.2)  # Peak price
        else:
            rtp[hour] = random.uniform(0.4, 0.6)  # Off-peak price
    return rtp

pricing_curve = generate_rtp()

# Schedule Non-Shiftable Appliances
non_shiftable_schedule = np.zeros((len(non_shiftable_appliances), num_hours))

for i, (appliance, details) in enumerate(non_shiftable_appliances.items()):
    start, end = details["start"], details["end"]
    non_shiftable_schedule[i, start:end] = 1  # Always ON during this period

# Optimize Scheduling for Shiftable Appliances
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
            penalty = 5 if t in peak_hours else 0  # Large penalty to avoid peak hours
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

# Run Optimization
result, shiftable_appliances_list = optimize_schedule()

# Output Results
if result.success:
    print(f"\nOptimal cost: {result.fun:.2f} NOK\n")

    # Extract scheduling decisions
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

    plt.xlabel("Hour of the Day")
    plt.ylabel("Power Usage (kWh)")
    plt.title("Optimized Appliance Scheduling vs. RTP Pricing")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()

else:
    print("Optimization failed.")