from django import forms

class PredictionForm(forms.Form):
    home_number = forms.CharField(
        label="Home Number",
        max_length=10,  # Adjust if needed
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'e.g., B-21',
            'class': 'form-control'  # Bootstrap class for styling
        })
    )
    date = forms.DateField(
        label="Date",
        required=True,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'form-control'  # Bootstrap class for styling
        })
    )
    time = forms.TimeField(
        label="Time",
        required=True,
        widget=forms.TimeInput(attrs={
            'type': 'time',
            'step': 60,  # Step = 60s for minute precision
            'class': 'form-control'  # Bootstrap class for styling
        })
    )