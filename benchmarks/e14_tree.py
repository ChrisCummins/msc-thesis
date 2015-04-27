def classify(x):
  if x["DevType"] == "GPU":
    # 70 correct / 14 incorrect, 83% accurate:
    if x["BorderEast"] <= "10": return (64, 4)
    if x["BorderEast"] > "10":
      # 0 correct / 0 incorrect:
      if x["Hostname"] == "cec": return (32, 32)
      # 8 correct / 0 incorrect, 100% accurate:
      if x["Hostname"] == "monza": return (64, 4)
      # 0 correct / 0 incorrect:
      if x["Hostname"] == "florence": return (32, 32)
      # 8 correct / 0 incorrect, 100% accurate:
      if x["Hostname"] == "whz5": return (32, 32)
      if x["Hostname"] == "tim":
        if x["Complexity"] == "0":
          # 4 correct / 0 incorrect, 100% accurate:
          if x["BorderNorth"] <= "20": return (64, 4)
          # 4 correct / 0 incorrect, 100% accurate:
          if x["BorderNorth"] > "20": return (32, 32)
        # 13 correct / 0 incorrect, 100% accurate:
        if x["Complexity"] == "1": return (32, 32)
  if x["DevType"] == "CPU":
    if x["BorderNorth"] <= "20":
      if x["BorderSouth"] <= "10":
        if x["Hostname"] == "cec":
          # 4 correct / 0 incorrect, 100% accurate:
          if x["BorderNorth"] <= "1": return (64, 4)
          # 8 correct / 5 incorrect, 62% accurate:
          if x["BorderNorth"] > "1": return (64, 32)
        # 6 correct / 0 incorrect, 100% accurate:
        if x["Hostname"] == "monza": return (32, 32)
        if x["Hostname"] == "florence":
          # 6 correct / 2 incorrect, 75% accurate:
          if x["DataWidth"] <= "1024": return (64, 32)
          # 6 correct / 1 incorrect, 86% accurate:
          if x["DataWidth"] > "1024": return (64, 64)
        # 0 correct / 0 incorrect:
        if x["Hostname"] == "whz5": return (64, 32)
        # 0 correct / 0 incorrect:
        if x["Hostname"] == "tim": return (64, 32)
      if x["BorderSouth"] > "10":
        if x["Hostname"] == "cec":
          # 6 correct / 0 incorrect, 100% accurate:
          if x["Complexity"] == "0": return (32, 64)
          if x["Complexity"] == "1":
            # 2 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] <= "10": return (32, 4)
            if x["BorderNorth"] > "10":
              # 2 correct / 1 incorrect, 67% accurate:
              if x["BorderEast"] <= "10": return (32, 64)
              # 2 correct / 0 incorrect, 100% accurate:
              if x["BorderEast"] > "10": return (64, 32)
        # 4 correct / 1 incorrect, 80% accurate:
        if x["Hostname"] == "monza": return (4, 64)
        if x["Hostname"] == "florence":
          if x["Complexity"] == "0":
            # 2 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] <= "10": return (4, 64)
            if x["BorderNorth"] > "10":
              # 2 correct / 1 incorrect, 67% accurate:
              if x["BorderEast"] <= "10": return (64, 32)
              # 2 correct / 0 incorrect, 100% accurate:
              if x["BorderEast"] > "10": return (4, 32)
          # 6 correct / 1 incorrect, 86% accurate:
          if x["Complexity"] == "1": return (32, 64)
        # 0 correct / 0 incorrect:
        if x["Hostname"] == "whz5": return (32, 64)
        # 0 correct / 0 incorrect:
        if x["Hostname"] == "tim": return (32, 64)
    if x["BorderNorth"] > "20":
      # 4 correct / 0 incorrect, 100% accurate:
      if x["Hostname"] == "cec": return (64, 4)
      # 2 correct / 0 incorrect, 100% accurate:
      if x["Hostname"] == "monza": return (4, 64)
      # 4 correct / 0 incorrect, 100% accurate:
      if x["Hostname"] == "florence": return (64, 4)
      # 0 correct / 0 incorrect:
      if x["Hostname"] == "whz5": return (64, 4)
      # 0 correct / 0 incorrect:
      if x["Hostname"] == "tim": return (64, 4)
