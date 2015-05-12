def classify(x):
  if x["dev_denorms"] == "1":
    if x["dev_local_memory_size"] <= "49151":
      if x["BorderSouth"] <= "10":
        if x["dev_global_memory_size"] <= "8284794880":
          # 8 correct / 0 incorrect, 100% accurate:
          if x["BorderNorth"] <= "1": return (64, 4)
          if x["BorderNorth"] > "1":
            if x["BorderNorth"] <= "5":
              if x["host_freq"] <= "3401000":
                # 2 correct / 0 incorrect, 100% accurate:
                if x["kernel_instructions_of_all_types_"] <= "197": return (64, 4)
                # 2 correct / 0 incorrect, 100% accurate:
                if x["kernel_instructions_of_all_types_"] > "197": return (16, 16)
              if x["host_freq"] > "3401000":
                # 2 correct / 0 incorrect, 100% accurate:
                if x["kernel_instructions_of_all_types_"] <= "197": return (64, 16)
                # 2 correct / 1 incorrect, 67% accurate:
                if x["kernel_instructions_of_all_types_"] > "197": return (16, 32)
            # 8 correct / 1 incorrect, 89% accurate:
            if x["BorderNorth"] > "5": return (64, 16)
        if x["dev_global_memory_size"] > "8284794880":
          if x["host_freq"] <= "3000000":
            # 6 correct / 3 incorrect, 67% accurate:
            if x["DataWidth"] <= "1024": return (64, 16)
            # 6 correct / 1 incorrect, 86% accurate:
            if x["DataWidth"] > "1024": return (64, 64)
          if x["host_freq"] > "3000000":
            # 4 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] <= "5": return (64, 16)
            # 2 correct / 0 incorrect, 100% accurate:
            if x["BorderNorth"] > "5": return (32, 32)
      if x["BorderSouth"] > "10":
        if x["dev_global_memory_size"] <= "8284794880":
          if x["BorderNorth"] <= "10":
            # 2 correct / 0 incorrect, 100% accurate:
            if x["kernel_instructions_of_all_types_"] <= "197": return (16, 32)
            # 2 correct / 0 incorrect, 100% accurate:
            if x["kernel_instructions_of_all_types_"] > "197": return (16, 4)
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
                  # 2 correct / 1 incorrect, 67% accurate:
                  if x["kernel_instructions_of_all_types_"] <= "197": return (16, 16)
                  # 2 correct / 0 incorrect, 100% accurate:
                  if x["kernel_instructions_of_all_types_"] > "197": return (64, 16)
                # 4 correct / 1 incorrect, 80% accurate:
                if x["BorderNorth"] > "20": return (32, 16)
        if x["dev_global_memory_size"] > "8284794880":
          if x["BorderNorth"] <= "20":
            if x["kernel_instructions_of_all_types_"] <= "197":
              # 6 correct / 3 incorrect, 67% accurate:
              if x["BorderEast"] <= "10": return (4, 64)
              if x["BorderEast"] > "10":
                # 2 correct / 0 incorrect, 100% accurate:
                if x["host_freq"] <= "3000000": return (4, 32)
                # 2 correct / 0 incorrect, 100% accurate:
                if x["host_freq"] > "3000000": return (16, 64)
            # 6 correct / 1 incorrect, 86% accurate:
            if x["kernel_instructions_of_all_types_"] > "197": return (32, 64)
          if x["BorderNorth"] > "20":
            if x["host_freq"] <= "3000000":
              # 2 correct / 1 incorrect, 67% accurate:
              if x["DataWidth"] <= "1024": return (16, 32)
              # 2 correct / 0 incorrect, 100% accurate:
              if x["DataWidth"] > "1024": return (64, 4)
            # 2 correct / 0 incorrect, 100% accurate:
            if x["host_freq"] > "3000000": return (4, 64)
    if x["dev_local_memory_size"] > "49151":
      # 34 correct / 5 incorrect, 87% accurate:
      if x["BorderEast"] <= "10": return (64, 4)
      if x["BorderEast"] > "10":
        if x["kernel_instructions_of_all_types_"] <= "197":
          # 4 correct / 0 incorrect, 100% accurate:
          if x["BorderNorth"] <= "20": return (64, 4)
          # 4 correct / 0 incorrect, 100% accurate:
          if x["BorderNorth"] > "20": return (32, 32)
        # 13 correct / 0 incorrect, 100% accurate:
        if x["kernel_instructions_of_all_types_"] > "197": return (32, 32)
  if x["dev_denorms"] == "0":
    # 8 correct / 3 incorrect, 73% accurate:
    if x["BorderEast"] <= "5": return (64, 4)
    if x["BorderEast"] > "5":
      if x["BorderEast"] <= "20":
        if x["BorderSouth"] <= "10":
          # 2 correct / 0 incorrect, 100% accurate:
          if x["kernel_instructions_of_all_types_"] <= "197": return (64, 4)
          # 2 correct / 0 incorrect, 100% accurate:
          if x["kernel_instructions_of_all_types_"] > "197": return (16, 16)
        # 12 correct / 0 incorrect, 100% accurate:
        if x["BorderSouth"] > "10": return (16, 16)
      if x["BorderEast"] > "20":
        # 2 correct / 0 incorrect, 100% accurate:
        if x["kernel_instructions_of_all_types_"] <= "197": return (16, 16)
        # 2 correct / 0 incorrect, 100% accurate:
        if x["kernel_instructions_of_all_types_"] > "197": return (64, 4)
