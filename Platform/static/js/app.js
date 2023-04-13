
async function checkViolations(phrase) {
    // Hanyang put api here i think
    const apiUrl = "";
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ phrase }),
    });
  
    if (response.ok) {
      const data = await response.json();
      return data.violations;
    } else {
      throw new Error("Error fetching data from the API");
    }
  }
  
  // Function to update the icons based on the violations
  function updateIcons(violations) {
    const iconContainers = document.querySelectorAll(".icon-container");
  
    iconContainers.forEach((iconContainer) => {
      const id = iconContainer.id;
  
      if (violations.includes(id)) {
        iconContainer.classList.add("violated");
      } else {
        iconContainer.classList.remove("violated");
      }
    });
  }
  
  // Event listener for the "Check Phrase" button
  document.getElementById("check-button").addEventListener("click", async () => {
    const phrase = document.getElementById("phrase-input").value;
  
    try {
      const violations = await checkViolations(phrase);
      updateIcons(violations);
    } catch (error) {
      console.error("Error:", error.message);
    }
  });