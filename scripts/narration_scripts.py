"""Narration scripts for the Ocean IQ faceless SIH video.

Two versions:
  V1  = the script exactly as written (safe, zero hard numbers)
  V2  = same structure, with verified measured metrics injected

Every value in V2 is a real measurement taken from this repository:
  PHASE2_RESULTS.md, PHASE4_5_RESULTS.md, PHASE8_MISSION_VALIDATION.md
"""

V1 = {
"1. Hook + Problem": [
    "Beneath the ocean surface, dangerous debris can remain hidden inside vast areas of seabed.",
    "Ghost nets and other man-made debris can threaten marine life, damage ecosystems, and create hazards for vessels.",
    "The problem becomes even harder with side-scan sonar.",
    "A single survey can contain huge amounts of acoustic imagery, making manual inspection slow and difficult.",
],

"2. Solution": [
    "This is the problem Ocean IQ is designed to address.",
    "Ocean IQ transforms raw side-scan sonar surveys into structured underwater intelligence.",
    "The system first analyzes and enhances sonar imagery.",
    "It then uses artificial intelligence to identify potential man-made debris and underwater anomalies.",
    "But detection alone is not enough.",
    "The system also evaluates confidence and acoustic evidence around the detected object.",
    "Sonar geometry can then be used for spatial information when the required data is available.",
    "Finally, the results are presented through an operator dashboard and reporting system.",
    "The goal is simple.",
    "Turn large volumes of sonar imagery into information that an operator can actually use.",
],

"3. Architecture": [
    "Now, the complete system can be understood as a journey of sonar data.",
    "It begins with side-scan sonar data collected during a marine survey.",
    "The sonar imagery enters the preprocessing stage.",
    "Noise is reduced, contrast is enhanced, and the imagery is prepared for A.I. analysis.",
    "The next stage is object detection and classification.",
    "A YOLO based model identifies potential underwater objects from the processed sonar imagery.",
    "After detection, the system performs additional validation.",
    "Confidence scoring and acoustic shadow analysis provide additional evidence around the detected contact.",
    "The next stage is localization.",
    "Sonar geometry and W G S 84 georeferencing convert available detection information into spatial information.",
    "Finally, the processed information reaches the application layer.",
    "The backend handles processing and inference logic.",
    "The frontend presents detections, locations, object information, and reports to the operator.",
    "The physical layer remains the side-scan sonar payload carried by an A U V or a towed survey system.",
    "This architecture connects sonar acquisition, artificial intelligence, physical analysis, localization, and visualization into one workflow.",
],

"4. Live Demo": [
    "Now let's see the system in action.",
    "The deployed application provides a tactical interface for working with sonar detections and mission information.",
    "The process begins with a side-scan sonar image.",
    "The sonar image enters the analysis pipeline and is processed for object detection.",
    "The detector analyzes the sonar imagery and identifies potential underwater contacts.",
    "Each detection can be displayed with its class and confidence information.",
    "The important point is that the output remains connected to the original sonar evidence.",
    "The Detection Inspector provides additional information about the contact.",
    "It can show the track identifier, object class, persistence information, observation count, track age, and confidence history.",
    "The system also keeps track of where every measurement comes from.",
    "Values can be marked as measured, derived, assumed, simulated, demo, or unavailable.",
    "This distinction is important, because missing sensor information should not become a fabricated measurement.",
    "The System One reflex layer provides fast, deterministic operational decisions.",
    "It can classify the contact into actions such as passive logging, loiter and rescan, or emergency prop hazard.",
    "These decisions are advisory only.",
    "The system does not control the vehicle.",
    "A second reasoning layer provides higher level tactical interpretation.",
    "This information can be presented as a briefing for operators and mission teams.",
    "Finally, the detection can be connected to the mission timeline and geospatial view.",
    "The system can also generate a structured P D F report for operational use.",
    "And the test does not stop with positive detections.",
    "A clean seabed sample is also passed through the detector.",
    "In this test, the system returns zero detections.",
],

"5. Impact + Closing": [
    "The objective is to reduce the effort required to inspect large sonar surveys.",
    "Faster detection can support marine survey operations, underwater hazard identification, and marine cleanup activities.",
    "The system combines artificial intelligence, sonar analysis, localization, and operational reporting in one workflow.",
    "Ocean IQ turns raw sonar imagery into structured underwater intelligence.",
    "This is the approach developed for Smart India Hackathon twenty twenty six, Problem Statement twenty six thousand fifty seven.",
],
}


V2 = {
"1. Hook + Problem": [
    "Beneath the ocean surface, dangerous debris can remain hidden across vast areas of the seabed.",
    "Ghost nets, and other abandoned fishing gear, trap and kill endangered marine life.",
    "They damage coral reefs, and foul the propellers of vessels that pass overhead.",
    "Now here is the difficult part.",
    "Below roughly twenty meters of depth, light simply does not reach.",
    "An optical camera is blind. So underwater vessels use side-scan sonar instead.",
    "But a single survey can return hundreds of kilometers of acoustic imagery.",
    "Inspecting it manually takes hydrographers weeks, or even months.",
],

"2. Solution": [
    "This is the problem Ocean IQ is designed to address.",
    "Ocean IQ transforms raw side-scan sonar surveys into structured underwater intelligence.",
    "The system first analyzes and enhances the sonar imagery.",
    "It then uses artificial intelligence to identify man-made debris and underwater anomalies.",
    "The detector recognizes five classes.",
    "Ghost nets, shipwrecks, submarine pipelines, cylindrical mines, and crab pots.",
    "But detection alone is not enough.",
    "The system also evaluates confidence and acoustic evidence around each detected object.",
    "And here is the part that matters.",
    "Sonar geometry is used to derive the physical height of the object above the seabed.",
    "Finally, everything is presented through an operator dashboard and reporting system.",
    "The goal is simple.",
    "Turn large volumes of sonar imagery into information an operator can actually use.",
],

"3. Architecture": [
    "Now, the complete system can be understood as a journey of sonar data.",
    "It begins with side-scan sonar data collected during a marine survey.",
    "The imagery enters preprocessing, where contrast limited adaptive histogram equalization stretches the acoustic detail, and a bilateral filter removes speckle noise while preserving sharp shadow edges.",
    "The next stage is detection.",
    "A YOLO based model, exported to O N N X, identifies the objects.",
    "Why O N N X?",
    "Because the original weights were one and a half gigabytes.",
    "The exported model is just forty two point seven megabytes.",
    "That is what makes edge deployment on a vehicle possible.",
    "The imagery is processed in overlapping tiles of six hundred and forty pixels, because real sonar strips are twenty thousand pixels long, and resizing them would destroy small nets.",
    "After detection, the system performs acoustic shadow analysis.",
    "The length of the shadow behind an object, combined with the sonar altitude, gives its elevation in meters.",
    "Then comes localization, using W G S 84 georeferencing.",
    "A temporal tracking layer compares repeated survey passes, so a contact seen twice is upgraded from a new contact to confirmed debris.",
    "That filters out fish, and other moving marine life.",
    "The decisions themselves run across two layers.",
    "System One is a fast reflex engine that runs entirely offline on the vehicle.",
    "It returns one of three actions, under a hundred milliseconds.",
    "System Two is a slower reasoning layer, at the surface station, that produces a tactical briefing.",
    "Finally, the results reach the application layer.",
    "A Python backend handles processing, and a dependency free frontend presents everything to the operator.",
    "The physical layer remains the side-scan sonar payload, carried by an A U V or a towed survey system.",
    "And one safety note, which we built in deliberately.",
    "The system is advisory only. It never controls the vehicle.",
],

"4. Live Demo": [
    "Now let's see the system in action.",
    "This is the operator cockpit.",
    "The process begins with a side-scan sonar image.",
    "Let's run a ghost net sample through the detector.",
    "The model returns a ghost net contact, at ninety two point nine percent confidence, in one hundred and ninety three milliseconds, on an ordinary C P U.",
    "The bounding box appears on the sonar waterfall viewer.",
    "The Detection Inspector opens with the track identifier, object class, and a confidence history sparkline.",
    "The System One reflex layer updates with the decision primitive and a hazard score of eight point five out of ten, and escalates it to emergency prop hazard.",
    "The System Two layer produces a tactical briefing for the operator.",
    "Now here is the detail we care most about.",
    "This system keeps track of where every single measurement comes from.",
    "Elevation is labelled as derived.",
    "Shadow length is labelled as measured.",
    "Missing G P S is labelled as unavailable, never invented.",
    "Because missing sensor information must never become a fabricated measurement.",
    "The detection is then connected to the mission timeline, and pinned on the geospatial view.",
    "One click exports a structured P D F recovery report for operational use.",
    "And the test does not stop with positive detections.",
    "This is a clean seabed sample, with nothing on it.",
    "Passed through the same detector, it returns zero detections.",
    "Zero false positives.",
    "That is the number that proves the system is trustworthy.",
],

"5. Impact + Closing": [
    "The objective is to reduce the effort required to inspect large sonar surveys.",
    "Faster detection can support marine survey operations, underwater hazard identification, and marine cleanup activities.",
    "The system combines artificial intelligence, sonar analysis, localization, and operational reporting, in one workflow.",
    "It runs on a five hundred and twelve megabyte server, and the entire test suite is fully automated.",
    "Ocean IQ turns raw sonar imagery into structured underwater intelligence.",
    "This is the approach developed for Smart India Hackathon twenty twenty six, Problem Statement twenty six thousand fifty seven.",
],
}
