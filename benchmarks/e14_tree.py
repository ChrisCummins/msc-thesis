def classify(x):
  if x["dev_max_number_of_images_read_arguments"] <= "128":
    if x["dev_denorms"] == "1":
      if x["host_freq"] <= "3401000":
        # 34 correct / 5 incorrect, 87% accurate:
        if x["BorderEast"] <= "10": return (64, 4)
        if x["BorderEast"] > "10":
          # 13 correct / 0 incorrect, 100% accurate:
          if x["Complexity"] == "1": return (32, 32)
          if x["Complexity"] == "0":
            # 4 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] <= "20": return (64, 4)
            # 4 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] > "20": return (32, 32)
      if x["host_freq"] > "3401000":
        if x["BorderNorth"] <= "10":
          # 3 correct / 1 incorrect, 75% accurate:
          if x["DataWidth"] <= "1024": return (64, 16)
          # 3 correct / 0 incorrect, 100% accurate:
          if x["DataWidth"] > "1024": return (32, 32)
        # 6 correct / 2 incorrect, 75% accurate:
        if x["BorderNorth"] > "10": return (4, 64)
    if x["dev_denorms"] == "0":
      # 8 correct / 3 incorrect, 73% accurate:
      if x["BorderEast"] <= "5": return (64, 4)
      if x["BorderEast"] > "5":
        # 10 correct / 2 incorrect, 83% accurate:
        if x["DataWidth"] <= "1024": return (16, 16)
        if x["DataWidth"] > "1024":
          # 5 correct / 1 incorrect, 83% accurate:
          if x["Complexity"] == "1": return (16, 16)
          # 5 correct / 0 incorrect, 100% accurate:
          if x["Complexity"] == "0": return (64, 4)
  if x["dev_max_number_of_images_read_arguments"] > "128":
    if x["BorderSouth"] <= "10":
      if x["host_freq"] <= "3000000":
        # 6 correct / 3 incorrect, 67% accurate:
        if x["DataWidth"] <= "1024": return (64, 16)
        # 6 correct / 1 incorrect, 86% accurate:
        if x["DataWidth"] > "1024": return (64, 64)
      if x["host_freq"] > "3000000":
        # 8 correct / 0 incorrect, 100% accurate:
        if x["BorderNorth"] <= "1": return (64, 4)
        if x["BorderNorth"] > "1":
          if x["BorderNorth"] <= "5":
            if x["host_freq"] <= "3401000":
              # 2 correct / 0 incorrect, 100% accurate:
              if x["Complexity"] == "1": return (16, 16)
              # 2 correct / 0 incorrect, 100% accurate:
              if x["Complexity"] == "0": return (64, 4)
            if x["host_freq"] > "3401000":
              # 2 correct / 1 incorrect, 67% accurate:
              if x["Complexity"] == "1": return (16, 32)
              # 2 correct / 0 incorrect, 100% accurate:
              if x["Complexity"] == "0": return (64, 16)
          # 8 correct / 1 incorrect, 89% accurate:
          if x["BorderNorth"] > "5": return (64, 16)
    if x["BorderSouth"] > "10":
      if x["host_freq"] <= "3000000":
        if x["BorderNorth"] <= "20":
          # 6 correct / 1 incorrect, 86% accurate:
          if x["Complexity"] == "1": return (32, 64)
          if x["Complexity"] == "0":
            # 2 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] <= "10": return (4, 64)
            if x["BorderNorth"] > "10":
              # 2 correct / 1 incorrect, 67% accurate:
              if x["BorderEast"] <= "10": return (64, 32)
              # 2 correct / 0 incorrect, 100% accurate:
              if x["BorderEast"] > "10": return (4, 32)
        if x["BorderNorth"] > "20":
          # 2 correct / 1 incorrect, 67% accurate:
          if x["DataWidth"] <= "1024": return (16, 32)
          # 2 correct / 0 incorrect, 100% accurate:
          if x["DataWidth"] > "1024": return (64, 4)
      if x["host_freq"] > "3000000":
        if x["BorderNorth"] <= "10":
          # 2 correct / 0 incorrect, 100% accurate:
          if x["Complexity"] == "1": return (16, 4)
          # 2 correct / 0 incorrect, 100% accurate:
          if x["Complexity"] == "0": return (16, 32)
        if x["BorderNorth"] > "10":
          if x["BorderEast"] <= "10":
            # 4 correct / 0 incorrect, 100% accurate:
            if x["host_freq"] <= "3401000": return (32, 16)
            # 4 correct / 1 incorrect, 80% accurate:
            if x["host_freq"] > "3401000": return (32, 64)
          if x["BorderEast"] > "10":
            if x["host_freq"] <= "3401000":
              # 4 correct / 1 incorrect, 80% accurate:
              if x["BorderNorth"] <= "20": return (32, 16)
              # 4 correct / 0 incorrect, 100% accurate:
              if x["BorderNorth"] > "20": return (64, 16)
            if x["host_freq"] > "3401000":
              if x["BorderNorth"] <= "20":
                # 2 correct / 0 incorrect, 100% accurate:
                if x["Complexity"] == "1": return (64, 16)
                # 2 correct / 1 incorrect, 67% accurate:
                if x["Complexity"] == "0": return (16, 16)
              # 4 correct / 1 incorrect, 80% accurate:
              if x["BorderNorth"] > "20": return (32, 16)
