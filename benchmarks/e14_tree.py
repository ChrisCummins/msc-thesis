def classify(x):
  if x["host_nproc"] <= "4":
    if x["host_freq"] <= "3000000":
      if x["BorderNorth"] <= "20":
        if x["BorderWest"] <= "10":
          if x["DataWidth"] <= "1024":
            # 4 correct / 0 incorrect, 100% accurate:
            if x["Complexity"] == "0": return (64, 32)
            if x["Complexity"] == "1":
              # 2 correct / 1 incorrect, 67% accurate:
              if x["BorderNorth"] <= "5": return (64, 32)
              # 2 correct / 0 incorrect, 100% accurate:
              if x["BorderNorth"] > "5": return (32, 32)
          if x["DataWidth"] > "1024":
            # 6 correct / 1 incorrect, 86% accurate:
            if x["BorderNorth"] <= "10": return (64, 64)
            # 2 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] > "10": return (32, 64)
        if x["BorderWest"] > "10":
          if x["Complexity"] == "0":
            # 2 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] <= "10": return (4, 64)
            # 2 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] > "10": return (4, 32)
          # 4 correct / 0 incorrect, 100% accurate:
          if x["Complexity"] == "1": return (32, 64)
      # 4 correct / 0 incorrect, 100% accurate:
      if x["BorderNorth"] > "20": return (64, 4)
    if x["host_freq"] > "3000000":
      # 4 correct / 0 incorrect, 100% accurate:
      if x["BorderEast"] <= "1": return (64, 4)
      if x["BorderEast"] > "1":
        if x["BorderEast"] <= "20":
          if x["BorderNorth"] <= "1":
            # 2 correct / 0 incorrect, 100% accurate:
            if x["Complexity"] == "0": return (16, 32)
            # 2 correct / 0 incorrect, 100% accurate:
            if x["Complexity"] == "1": return (16, 4)
          if x["BorderNorth"] > "1":
            # 8 correct / 3 incorrect, 73% accurate:
            if x["BorderNorth"] <= "10": return (64, 16)
            if x["BorderNorth"] > "10":
              # 4 correct / 1 incorrect, 80% accurate:
              if x["BorderEast"] <= "10": return (32, 64)
              if x["BorderEast"] > "10":
                # 2 correct / 1 incorrect, 67% accurate:
                if x["Complexity"] == "0": return (16, 16)
                # 2 correct / 0 incorrect, 100% accurate:
                if x["Complexity"] == "1": return (64, 16)
        # 4 correct / 1 incorrect, 80% accurate:
        if x["BorderEast"] > "20": return (32, 16)
  if x["host_nproc"] > "4":
    if x["dev_address_bits"] <= "32":
      # 70 correct / 14 incorrect, 83% accurate:
      if x["BorderEast"] <= "10": return (64, 4)
      if x["BorderEast"] > "10":
        if x["host_freq"] <= "3401000":
          if x["Complexity"] == "0":
            if x["BorderNorth"] <= "20":
              # 4 correct / 0 incorrect, 100% accurate:
              if x["host_mem"] <= "8158500": return (64, 4)
              # 2 correct / 0 incorrect, 100% accurate:
              if x["host_mem"] > "8158500": return (32, 32)
            # 6 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] > "20": return (32, 32)
          # 17 correct / 0 incorrect, 100% accurate:
          if x["Complexity"] == "1": return (32, 32)
        # 8 correct / 0 incorrect, 100% accurate:
        if x["host_freq"] > "3401000": return (64, 4)
    if x["dev_address_bits"] > "32":
      # 8 correct / 1 incorrect, 89% accurate:
      if x["BorderEast"] <= "10": return (32, 32)
      # 4 correct / 0 incorrect, 100% accurate:
      if x["BorderEast"] > "10": return (4, 64)
