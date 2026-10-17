# PolyRoute interface refresh

## Plan
- Replace the blank home screen with a polished, responsive PolyRoute waste-routing workspace based on the supplied HTML and CSS.
- Reorganize the batch, weight, and facility controls into a clearer entry flow, with “Find routes” immediately below the inputs.
- Add a useful initial results preview and make weight-mode selection and route calculation respond to form input.
- Define the page palette and typography as semantic design-system tokens, and give the home page distinct metadata.
- Verify the page in the running preview and check build diagnostics.

## Technical details
- Implement in the TanStack Start home route with local React state; no external service or persistence is present in this project.
