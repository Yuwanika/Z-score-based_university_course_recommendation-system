document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('z-score-form');

    form.addEventListener('submit', function (e) {
        e.preventDefault(); // ⛔ Prevent page refresh

        // ✅ Get form values
        const district = document.getElementById('district').value;
        const zScore = document.getElementById('z-score').value;
        const stream = document.getElementById('stream').value;
        const year = document.getElementById('year').value;

        // ✅ Create JSON object
        const formData = {
            district: district,
            z_score: zScore,
            stream: stream,
            year: year
        };

        // ✅ Log JSON data
        console.log('Form Data (JSON):', JSON.stringify(formData, null, 2));

        // ✅ (Optional) Send to backend
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(res => res.json())
        .then(data => {
            console.log('Server response:', data);

            const courseData = data;
            const uni = courseData.university;
            const course = courseData.course;

            console.log('Selected University:', uni);
            console.log('Selected Course:', course);
            
        })

        .catch(err => {
            console.error('Error:', err);
        });
        
    });
});
console.log('Script loaded successfully');

/*
document.getElementById('frame').innerHTML = `
                <h1>Congratulations!</h1>
                <br>
                <h2>You have been selected for the</h2>
                <br>
                <br>
                <h2 id="selected-course">${data.course}</h2>
                <br>
                <h2 id="selected-university">${data.university}</h2>
                <br>
                <br>
                <br>
                <br>
                <p>This is an AI generated output. Please verify all information before making decisions.</p>
                <br>
                <button class="btn btn-secondary" onclick="window.location.href='index.html'">Go Back</button>
            `
*/