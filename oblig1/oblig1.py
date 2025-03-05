import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import linprog

num_hours = 24
# Appliance data (power consumption per hour in kWh)
appliances = {
    # Non-shiftable appliances
    "lighting": {"power": 1.5, "start": 10, "end": 20},  # average for Lighting
    "heating": {"power": 7.0, "start": 0, "end": 24},  # heating is flexible
    "refrigerator": {"power": 1.32, "start": 0, "end": 24},  # refrigerator, always on
    "electric_stove": {"power": 3.9, "start": 0, "end": 24},
    "tv": {"power": 0.375, "start": 0, "end": 24},  # average power
    "computer": {"power": 0.6, "start": 0, "end": 24},  # assume one computer
    # shiftable appliances
    "dishwasher": {"power": 1.44, "start": 0, "end": 24},
    "washing_machine": {"power": 1.94, "start": 0, "end": 24},
    "cloth_dryer": {"power": 2.5, "start": 0, "end": 24},
    "ev": {"power": 9.9, "start": 0, "end": 24},
    # extra appliances
    "coffee_maker": {"power": 0.6, "start": 0, "end": 24},
    "ceiling_fan": {"power": 0.12, "start": 0, "end": 24},
    "hair_dryer": {"power": 1.5, "start": 0, "end": 24},
    "toaster": {"power": 0.6, "start": 0, "end": 24},
    "microwave": {"power": 0.9, "start": 0, "end": 24},
    "router": {"power": 0.05, "start": 0, "end": 24},
    "cellphone_charger": {"power": 0.05, "start": 0, "end": 24},
    "cloth_iron": {"power": 1.2, "start": 0, "end": 24},
    "freezer": {"power": 1.32, "start": 0, "end": 24}
}

# Generate random RTP pricing curve
def generate_rtp():
    rtp = np.zeros(num_hours)
    peak_hours = [17, 18, 19]  # Peak hours: 5 PM - 8 PM
    for hour in range(num_hours):
        if hour in peak_hours:
            rtp[hour] = random.uniform(0.8, 1.2)  # Peak price
        else:
            rtp[hour] = random.uniform(0.4, 0.6)  # Off-peak price
    return rtp

pricing_curve = generate_rtp()

# Visualize the pricing curve
plt.plot(range(num_hours), pricing_curve, label="RTP Pricing Curve")
plt.xlabel("Hour of the Day")
plt.ylabel("Cost per kWh (NOK)")
plt.title("Real-Time Pricing (RTP) Curve")
plt.legend()
plt.show()

# **OPTIMIZATION FUNCTION**
def optimize_schedule():
    appliances_list = list(appliances.keys())
    num_appliances = len(appliances_list)

    # **Decision Variables: x_ij where**
    # - i = appliance index
    # - j = hour of the day
    # Total number of decision variables = num_appliances * num_hours
    c = []  # Cost coefficients
    A_eq = []
    b_eq = []

    # Build cost vector
    for appliance in appliances_list:
        power = appliances[appliance]["power"]
        for t in range(num_hours):
            c.append(power * pricing_curve[t])  # Cost for using this appliance at this hour

    # Build equality constraints
    for i, appliance in enumerate(appliances_list):
        power = appliances[appliance]["power"]
        start, end = appliances[appliance]["start"], appliances[appliance]["end"]

        # Constraint: Appliance should run for its required energy level
        constraint_row = [0] * (num_appliances * num_hours)  # Full zero row
        for t in range(start, end):  # Only allow usage in its valid hours
            constraint_row[i * num_hours + t] = 1  # Select variables for this appliance

        A_eq.append(constraint_row)
        b_eq.append(power)  # Required energy per appliance

    # Convert lists to NumPy arrays
    A_eq = np.array(A_eq)
    c = np.array(c)
    b_eq = np.array(b_eq)

    # Check matrix shapes
    print(f"Shape of A_eq: {A_eq.shape}")
    print(f"Length of c: {len(c)}")
    print(f"Length of b_eq: {len(b_eq)}")

    # Solve using linear programming
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * len(c), method='highs')

    return result

# **RUN OPTIMIZATION**
result = optimize_schedule()

# Check optimization results
if result.success:
    print(f"Optimal cost: {result.fun} NOK")
else:
    print("Optimization failed.")