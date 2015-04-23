def classify(x):
  if x["DevType"] == "GPU":
    if x["BorderEast"] <= "10": return (64, 4)
    if x["BorderEast"] > "10":
      if x["Hostname"] == "cec": return (32, 32)
      if x["Hostname"] == "monza": return (64, 4)
      if x["Hostname"] == "florence": return (32, 32)
      if x["Hostname"] == "whz5": return (32, 32)
      if x["Hostname"] == "tim":
        if x["Complexity"] == "0":
          if x["BorderNorth"] <= "20": return (64, 4)
          if x["BorderNorth"] > "20": return (32, 32)
        if x["Complexity"] == "1": return (32, 32)
  if x["DevType"] == "CPU":
    if x["BorderNorth"] <= "20":
      if x["BorderSouth"] <= "10":
        if x["Hostname"] == "cec":
          if x["BorderNorth"] <= "1": return (64, 4)
          if x["BorderNorth"] > "1": return (64, 32)
        if x["Hostname"] == "monza": return (32, 32)
        if x["Hostname"] == "florence":
          if x["DataWidth"] <= "1024": return (64, 32)
          if x["DataWidth"] > "1024": return (64, 64)
        if x["Hostname"] == "whz5": return (64, 32)
        if x["Hostname"] == "tim": return (64, 32)
      if x["BorderSouth"] > "10":
        if x["Hostname"] == "cec":
          if x["Complexity"] == "0": return (32, 64)
          if x["Complexity"] == "1":
            if x["BorderNorth"] <= "10": return (32, 4)
            if x["BorderNorth"] > "10":
              if x["BorderEast"] <= "10": return (32, 64)
              if x["BorderEast"] > "10": return (64, 32)
        if x["Hostname"] == "monza": return (4, 64)
        if x["Hostname"] == "florence":
          if x["Complexity"] == "0":
            if x["BorderNorth"] <= "10": return (4, 64)
            if x["BorderNorth"] > "10":
              if x["BorderEast"] <= "10": return (64, 32)
              if x["BorderEast"] > "10": return (4, 32)
          if x["Complexity"] == "1": return (32, 64)
        if x["Hostname"] == "whz5": return (32, 64)
        if x["Hostname"] == "tim": return (32, 64)
    if x["BorderNorth"] > "20":
      if x["Hostname"] == "cec": return (64, 4)
      if x["Hostname"] == "monza": return (4, 64)
      if x["Hostname"] == "florence": return (64, 4)
      if x["Hostname"] == "whz5": return (64, 4)
      if x["Hostname"] == "tim": return (64, 4)
