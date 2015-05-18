def classify(x):
  if x["dev_max_number_of_images_read_arguments"] <= "128":
    if x["dev_address_bits"] <= "32":
      if x["dev_alignment_bits_of_base_address"] <= "2048":
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
      if x["dev_alignment_bits_of_base_address"] > "2048":
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
    if x["dev_address_bits"] > "32":
      # 4 correct / 0 incorrect, 100% accurate:
      if x["BorderNorth"] <= "5": return (64, 16)
      if x["BorderNorth"] > "5":
        # 4 correct / 1 incorrect, 80% accurate:
        if x["BorderEast"] <= "10": return (32, 32)
        if x["BorderEast"] > "10":
          # 2 correct / 0 incorrect, 100% accurate:
          if x["BorderNorth"] <= "20": return (16, 64)
          # 2 correct / 0 incorrect, 100% accurate:
          if x["BorderNorth"] > "20": return (4, 64)
  if x["dev_max_number_of_images_read_arguments"] > "128":
    if x["dev_address_bits"] <= "32":
      # 8 correct / 1 incorrect, 89% accurate:
      if x["BorderNorth"] <= "5": return (48, 4)
      if x["BorderNorth"] > "5":
        # 8 correct / 1 incorrect, 89% accurate:
        if x["BorderEast"] <= "10": return (64, 8)
        if x["BorderEast"] > "10":
          # 4 correct / 1 incorrect, 80% accurate:
          if x["BorderNorth"] <= "20": return (32, 16)
          # 4 correct / 0 incorrect, 100% accurate:
          if x["BorderNorth"] > "20": return (64, 16)
    if x["dev_address_bits"] > "32":
      if x["BorderSouth"] <= "20":
        if x["kernel_instructions_of_all_types_"] <= "197":
          if x["dev_global_memory_size"] <= "8284794880":
            # 4 correct / 1 incorrect, 80% accurate:
            if x["BorderNorth"] <= "5": return (64, 8)
            if x["BorderNorth"] > "5":
              # 3 correct / 1 incorrect, 75% accurate:
              if x["DataWidth"] <= "1024": return (8, 24)
              # 3 correct / 1 incorrect, 75% accurate:
              if x["DataWidth"] > "1024": return (8, 48)
          if x["dev_global_memory_size"] > "8284794880":
            # 2 correct / 1 incorrect, 67% accurate:
            if x["BorderNorth"] <= "1": return (48, 24)
            if x["BorderNorth"] > "1":
              # 6 correct / 2 incorrect, 75% accurate:
              if x["BorderEast"] <= "10": return (64, 48)
              # 2 correct / 0 incorrect, 100% accurate:
              if x["BorderEast"] > "10": return (4, 24)
        if x["kernel_instructions_of_all_types_"] > "197":
          # 4 correct / 1 incorrect, 80% accurate:
          if x["BorderNorth"] <= "1": return (64, 8)
          if x["BorderNorth"] > "1":
            if x["dev_global_memory_size"] <= "8284794880":
              if x["BorderNorth"] <= "10":
                # 2 correct / 1 incorrect, 67% accurate:
                if x["BorderNorth"] <= "5": return (48, 48)
                # 2 correct / 0 incorrect, 100% accurate:
                if x["BorderNorth"] > "5": return (48, 24)
              if x["BorderNorth"] > "10":
                # 2 correct / 0 incorrect, 100% accurate:
                if x["BorderEast"] <= "10": return (24, 48)
                # 2 correct / 1 incorrect, 67% accurate:
                if x["BorderEast"] > "10": return (48, 48)
            # 8 correct / 6 incorrect, 57% accurate:
            if x["dev_global_memory_size"] > "8284794880": return (48, 64)
      if x["BorderSouth"] > "20":
        if x["kernel_instructions_of_all_types_"] <= "197":
          if x["dev_global_memory_size"] <= "8284794880":
            # 2 correct / 1 incorrect, 67% accurate:
            if x["BorderNorth"] <= "10": return (64, 24)
            # 2 correct / 1 incorrect, 67% accurate:
            if x["BorderNorth"] > "10": return (16, 16)
          if x["dev_global_memory_size"] > "8284794880":
            # 2 correct / 1 incorrect, 67% accurate:
            if x["BorderNorth"] <= "10": return (4, 16)
            # 2 correct / 1 incorrect, 67% accurate:
            if x["BorderNorth"] > "10": return (16, 32)
        if x["kernel_instructions_of_all_types_"] > "197":
          if x["BorderNorth"] <= "10":
            # 2 correct / 0 incorrect, 100% accurate:
            if x["dev_global_memory_size"] <= "8284794880": return (16, 4)
            # 2 correct / 1 incorrect, 67% accurate:
            if x["dev_global_memory_size"] > "8284794880": return (32, 64)
          # 4 correct / 1 incorrect, 80% accurate:
          if x["BorderNorth"] > "10": return (32, 24)
